// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Port of the motion2 target planner spec (planners/target.py + se2_search):
//! cloud z-band -> 2D distance field -> SE(2) lattice Dijkstra -> shortcut
//! smoothing -> densified (x, y, yaw) path. Deterministic by construction.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

pub const Z_BAND: (f64, f64) = (0.05, 0.45);
pub const FINE: f64 = 0.05;
pub const PAD: f64 = 1.5;
const CELL: f64 = 0.12;
const YAW_BINS: usize = 16;
const OFFSET_STEP: f64 = 0.05;

/// Embodiment parameters, mirroring scenarios.Embodiment.
#[derive(Clone, Copy, Debug)]
pub struct Emb {
    pub length: f64,
    pub width: f64,
    pub center_off: f64,
    pub comfort: f64,
    pub precision: f64,
    pub strafe: f64,
    pub reverse: f64,
    pub yaw_w: f64,
}

impl Emb {
    pub fn go2() -> Self {
        Emb {
            length: 0.85,
            width: 0.31,
            center_off: -0.025,
            comfort: 0.4,
            precision: 0.05,
            strafe: 1.8,
            reverse: 1.5,
            yaw_w: 0.25,
        }
    }
}

/// np.arange(start, stop, step) for positive step.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    let n = if n > 0.0 { n as usize } else { 0 };
    (0..n).map(|k| start + k as f64 * step).collect()
}

/// Footprint sample points, dense enough that thin slats can't slip.
fn offsets(emb: &Emb) -> Vec<(f64, f64)> {
    let (hl, hw) = (emb.length / 2.0, emb.width / 2.0);
    let xs = arange(-hl, hl + OFFSET_STEP / 2.0, OFFSET_STEP);
    let ys = arange(-hw, hw + OFFSET_STEP / 2.0, OFFSET_STEP);
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &x in &xs {
        for &y in &ys {
            out.push((x + emb.center_off, y));
        }
    }
    out
}

/// IEEE-style remainder onto (-pi, pi]: shortest yaw arc.
fn rem_2pi(x: f64) -> f64 {
    x - (x / (2.0 * PI)).round() * (2.0 * PI)
}

/// Uniform-bucket point index for exact nearest-neighbor distance queries.
struct PointBuckets {
    b: f64,
    x0: f64,
    y0: f64,
    nx: i64,
    ny: i64,
    cells: Vec<Vec<(f64, f64)>>,
}

impl PointBuckets {
    fn new(pts: &[(f64, f64)]) -> Self {
        let b = 0.2;
        let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
        let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in pts {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let nx = (((x1 - x0) / b).floor() as i64 + 1).max(1);
        let ny = (((y1 - y0) / b).floor() as i64 + 1).max(1);
        let mut cells = vec![Vec::new(); (nx * ny) as usize];
        for &(x, y) in pts {
            let i = (((x - x0) / b).floor() as i64).clamp(0, nx - 1);
            let j = (((y - y0) / b).floor() as i64).clamp(0, ny - 1);
            cells[(i * ny + j) as usize].push((x, y));
        }
        PointBuckets {
            b,
            x0,
            y0,
            nx,
            ny,
            cells,
        }
    }

    /// Exact distance from (qx, qy) to the nearest indexed point.
    fn nearest(&self, qx: f64, qy: f64) -> f64 {
        let qi = ((qx - self.x0) / self.b).floor() as i64;
        let qj = ((qy - self.y0) / self.b).floor() as i64;
        let max_ring = qi
            .abs()
            .max((self.nx - 1 - qi).abs())
            .max(qj.abs())
            .max((self.ny - 1 - qj).abs());
        let mut best = f64::INFINITY;
        let scan = |i: i64, j: i64, best: &mut f64| {
            if i < 0 || j < 0 || i >= self.nx || j >= self.ny {
                return;
            }
            for &(px, py) in &self.cells[(i * self.ny + j) as usize] {
                let d = (px - qx).hypot(py - qy);
                if d < *best {
                    *best = d;
                }
            }
        };
        for r in 0..=max_ring {
            // Any point in ring r sits at least (r-1)*b away: safe to stop.
            if best <= ((r - 1).max(0) as f64) * self.b {
                break;
            }
            if r == 0 {
                scan(qi, qj, &mut best);
                continue;
            }
            for i in (qi - r)..=(qi + r) {
                scan(i, qj - r, &mut best);
                scan(i, qj + r, &mut best);
            }
            for j in (qj - r + 1)..(qj + r) {
                scan(qi - r, j, &mut best);
                scan(qi + r, j, &mut best);
            }
        }
        best
    }
}

/// The candidate's world model: fine 2D distance field over the working area.
pub struct World {
    fx0: f64,
    fy0: f64,
    nfx: usize,
    nfy: usize,
    sdf: Vec<f64>,
    /// Lattice bounds (before the fine grid's extra 0.6 skirt).
    pub bounds: (f64, f64, f64, f64),
}

impl World {
    // round_ties_even matches np.round / python round at exact .5 cell edges.
    fn lookup(&self, px: f64, py: f64) -> f64 {
        let i = (((px - self.fx0) / FINE).round_ties_even() as i64).clamp(0, self.nfx as i64 - 1)
            as usize;
        let j = (((py - self.fy0) / FINE).round_ties_even() as i64).clamp(0, self.nfy as i64 - 1)
            as usize;
        self.sdf[i * self.nfy + j]
    }
}

/// Slice the cloud to the body z band and bake the fine distance field.
pub fn build_world(points: &[[f64; 3]], pose: (f64, f64, f64), goal: (f64, f64)) -> World {
    let band: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p[2] > Z_BAND.0 && p[2] < Z_BAND.1)
        .map(|p| (p[0], p[1]))
        .collect();
    let mut xs = vec![pose.0, goal.0];
    let mut ys = vec![pose.1, goal.1];
    for &(x, y) in &band {
        xs.push(x);
        ys.push(y);
    }
    let fmin = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let fmax = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (x0, y0) = (fmin(&xs) - PAD, fmin(&ys) - PAD);
    let (x1, y1) = (fmax(&xs) + PAD, fmax(&ys) + PAD);
    let fgx = arange(x0 - 0.6, x1 + 0.6, FINE);
    let fgy = arange(y0 - 0.6, y1 + 0.6, FINE);
    let (nfx, nfy) = (fgx.len(), fgy.len());
    let sdf = if band.is_empty() {
        vec![f64::INFINITY; nfx * nfy]
    } else {
        let buckets = PointBuckets::new(&band);
        let mut sdf = vec![0.0; nfx * nfy];
        for i in 0..nfx {
            for j in 0..nfy {
                sdf[i * nfy + j] = buckets.nearest(fgx[i], fgy[j]);
            }
        }
        sdf
    };
    World {
        fx0: fgx[0],
        fy0: fgy[0],
        nfx,
        nfy,
        sdf,
        bounds: (x0, y0, x1, y1),
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

struct Move {
    di: i64,
    dj: i64,
    base: f64,
    mids: Vec<(i64, i64)>,
}

/// Min-heap node: cost, then (bin, i, j) for deterministic tie-breaking.
#[derive(PartialEq)]
struct Node {
    d: f64,
    b: usize,
    i: usize,
    j: usize,
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.total_cmp(&self.d)
            .then(o.b.cmp(&self.b))
            .then(o.i.cmp(&self.i))
            .then(o.j.cmp(&self.j))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// The SE(2) lattice search (scenarios.se2_search semantics): 16 yaw bins,
/// 16-direction moves with knight-midpoint checks, blend edges, gait costs,
/// comfort-progressive clearance cost, validity-preserving shortcutting.
/// Returns sparse (x, y, yaw) vertices, or None when sealed.
pub fn se2_search(
    w: &World,
    start: (f64, f64, f64),
    goal: (f64, f64),
    emb: &Emb,
    margin: f64,
) -> Option<Vec<[f64; 3]>> {
    let (x0, y0, x1, y1) = w.bounds;
    let offs = offsets(emb);
    let gx = arange(x0, x1 + CELL, CELL);
    let gy = arange(y0, y1 + CELL, CELL);
    let (nx, ny) = (gx.len(), gy.len());
    let thetas: Vec<f64> = (0..YAW_BINS)
        .map(|k| -PI + k as f64 * (2.0 * PI / YAW_BINS as f64))
        .collect();
    let idx = |b: usize, i: usize, j: usize| (b * nx + i) * ny + j;

    // Body clearance per (yaw bin, cell): min field value over the footprint.
    let mut clr = vec![0.0f64; YAW_BINS * nx * ny];
    for (bi, &th) in thetas.iter().enumerate() {
        let (s, c) = th.sin_cos();
        let rot: Vec<(f64, f64)> = offs
            .iter()
            .map(|&(ox, oy)| (c * ox - s * oy, s * ox + c * oy))
            .collect();
        for i in 0..nx {
            for j in 0..ny {
                let mut m = f64::INFINITY;
                for &(rx, ry) in &rot {
                    let d = w.lookup(gx[i] + rx, gy[j] + ry);
                    if d < m {
                        m = d;
                    }
                }
                clr[idx(bi, i, j)] = m;
            }
        }
    }
    let free: Vec<bool> = clr.iter().map(|&v| v > margin).collect();
    // Comfort: clearance below emb.comfort is charged progressively, up to
    // ~2.5x at contact; beyond it, no charge.
    let pref = emb.comfort;
    let tight: Vec<f64> = clr
        .iter()
        .map(|&v| 1.0 + 1.5 * ((pref - v) / pref).clamp(0.0, 1.0))
        .collect();

    let cell_of = |px: f64, py: f64| -> (usize, usize) {
        (
            ((px - x0) / CELL)
                .round_ties_even()
                .clamp(0.0, (nx - 1) as f64) as usize,
            ((py - y0) / CELL)
                .round_ties_even()
                .clamp(0.0, (ny - 1) as f64) as usize,
        )
    };
    let mut sb = 0;
    let mut sbest = f64::INFINITY;
    for (k, &th) in thetas.iter().enumerate() {
        let d = rem_2pi(th - start.2).abs();
        if d < sbest {
            sbest = d;
            sb = k;
        }
    }
    let (si, sj) = cell_of(start.0, start.1);
    let (gi, gj) = cell_of(goal.0, goal.1);
    if !free[idx(sb, si, sj)] {
        return None;
    }

    // 16 directions: 8-connected + knight steps; knights check midpoint cells
    // so a thin wall cannot be hopped.
    let mut moves = Vec::new();
    for di in -2i64..=2 {
        for dj in -2i64..=2 {
            if (di, dj) == (0, 0) || gcd(di.abs(), dj.abs()) == 2 {
                continue;
            }
            let mids = if di.abs().max(dj.abs()) == 2 {
                vec![
                    (
                        (di as f64 / 2.0).floor() as i64,
                        (dj as f64 / 2.0).floor() as i64,
                    ),
                    (
                        (di as f64 / 2.0).ceil() as i64,
                        (dj as f64 / 2.0).ceil() as i64,
                    ),
                ]
            } else {
                Vec::new()
            };
            moves.push(Move {
                di,
                dj,
                base: ((di * di + dj * dj) as f64).sqrt() * CELL,
                mids,
            });
        }
    }
    // Gait-real costs: forward 1x, strafe/reverse scaled, yaw priced per rad.
    let move_cost = |base: f64, di: i64, dj: i64, th: f64| -> f64 {
        let rel = (dj as f64).atan2(di as f64) - th;
        let (f, l) = (rel.cos(), rel.sin());
        base * (1.0 + (emb.strafe - 1.0) * l.abs() + if f < 0.0 { emb.reverse - 1.0 } else { 0.0 })
    };
    let yaw_cost = emb.yaw_w * (2.0 * PI / YAW_BINS as f64);

    let mut dist = vec![f64::INFINITY; YAW_BINS * nx * ny];
    let mut prev = vec![u32::MAX; YAW_BINS * nx * ny];
    let mut heap: BinaryHeap<Node> = BinaryHeap::new();
    dist[idx(sb, si, sj)] = 0.0;
    heap.push(Node {
        d: 0.0,
        b: sb,
        i: si,
        j: sj,
    });
    let mut goal_state: Option<(usize, usize, usize)> = None;
    while let Some(Node { d, b, i, j }) = heap.pop() {
        if d > dist[idx(b, i, j)] {
            continue;
        }
        if (i, j) == (gi, gj) {
            goal_state = Some((b, i, j));
            break;
        }
        let try_move = |mv: &Move,
                        nb: usize,
                        extra: f64,
                        dist: &mut Vec<f64>,
                        prev: &mut Vec<u32>,
                        heap: &mut BinaryHeap<Node>| {
            let (ni, nj) = (i as i64 + mv.di, j as i64 + mv.dj);
            if ni < 0 || nj < 0 || ni >= nx as i64 || nj >= ny as i64 {
                return;
            }
            let (ni, nj) = (ni as usize, nj as usize);
            if !free[idx(nb, ni, nj)] {
                return;
            }
            if mv
                .mids
                .iter()
                .any(|&(mi, mj)| !free[idx(nb, (i as i64 + mi) as usize, (j as i64 + mj) as usize)])
            {
                return;
            }
            let c = (move_cost(mv.base, mv.di, mv.dj, thetas[b]) + extra) * tight[idx(nb, ni, nj)];
            if d + c < dist[idx(nb, ni, nj)] {
                dist[idx(nb, ni, nj)] = d + c;
                prev[idx(nb, ni, nj)] = idx(b, i, j) as u32;
                heap.push(Node {
                    d: d + c,
                    b: nb,
                    i: ni,
                    j: nj,
                });
            }
        };
        for mv in &moves {
            try_move(mv, b, 0.0, &mut dist, &mut prev, &mut heap);
        }
        for nb in [(b + 1) % YAW_BINS, (b + YAW_BINS - 1) % YAW_BINS] {
            let yc = yaw_cost * tight[idx(nb, i, j)];
            if free[idx(nb, i, j)] && d + yc < dist[idx(nb, i, j)] {
                dist[idx(nb, i, j)] = d + yc;
                prev[idx(nb, i, j)] = idx(b, i, j) as u32;
                heap.push(Node {
                    d: d + yc,
                    b: nb,
                    i,
                    j,
                });
            }
            // Blend edges: walk and turn in the same step, discounted.
            for mv in &moves {
                try_move(mv, nb, 0.5 * yaw_cost, &mut dist, &mut prev, &mut heap);
            }
        }
    }
    let (mut b, mut i, mut j) = goal_state?;
    let mut states = vec![(b, i, j)];
    while prev[idx(b, i, j)] != u32::MAX && (b, i, j) != (sb, si, sj) {
        let p = prev[idx(b, i, j)] as usize;
        b = p / (nx * ny);
        i = (p / ny) % nx;
        j = p % ny;
        states.push((b, i, j));
    }
    states.reverse();
    let raw: Vec<[f64; 3]> = states
        .iter()
        .map(|&(b, i, j)| [gx[i], gy[j], thetas[b]])
        .collect();

    // Shortcut smoothing, validity-preserving: a shortcut never reduces
    // clearance below what the raw detour had (capped at comfort).
    let pose_clear = |x: f64, y: f64, th: f64| -> f64 {
        let (s, c) = th.sin_cos();
        offs.iter()
            .map(|&(ox, oy)| w.lookup(x + c * ox - s * oy, y + s * ox + c * oy))
            .fold(f64::INFINITY, f64::min)
    };
    let seg_free = |a: &[f64; 3], b: &[f64; 3], floor: f64| -> bool {
        let dyaw = rem_2pi(b[2] - a[2]);
        let steps = 2usize
            .max(((b[0] - a[0]).hypot(b[1] - a[1]) / 0.06) as usize)
            .max((dyaw.abs() / 0.15) as usize);
        (0..=steps).all(|k| {
            let t = k as f64 / steps as f64;
            pose_clear(
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * dyaw,
            ) > floor
        })
    };
    let raw_clear: Vec<f64> = raw.iter().map(|s| pose_clear(s[0], s[1], s[2])).collect();
    let mut keep = vec![0usize];
    while *keep.last().unwrap() < raw.len() - 1 {
        let k = *keep.last().unwrap();
        let mut j = raw.len() - 1;
        while j > k + 1 {
            let minc = raw_clear[k..=j]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let floor = margin.max(minc.min(emb.comfort) - 0.02);
            if seg_free(&raw[k], &raw[j], floor) {
                break;
            }
            j -= 1;
        }
        keep.push(j);
    }
    Some(keep.iter().map(|&k| raw[k]).collect())
}

/// Interpolate sparse SE(2) vertices to path resolution (yaw = shortest arc).
pub fn densify(states: &[[f64; 3]], res: f64) -> Vec<[f64; 3]> {
    let mut dense = vec![states[0]];
    for w in states.windows(2) {
        let (a, b) = (w[0], w[1]);
        let dyaw = rem_2pi(b[2] - a[2]);
        let n = 1usize
            .max(((b[0] - a[0]).hypot(b[1] - a[1]) / res) as usize)
            .max((dyaw.abs() / 0.15) as usize);
        for k in 1..=n {
            let t = k as f64 / n as f64;
            dense.push([
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * dyaw,
            ]);
        }
    }
    dense
}

/// One plan call: world model from the cloud, SE(2) search, densified path.
/// None = refuse (no route the body can take).
pub fn plan(
    points: &[[f64; 3]],
    pose: (f64, f64, f64),
    goal: (f64, f64),
    emb: &Emb,
    resolution: f64,
) -> Option<Vec<[f64; 3]>> {
    let w = build_world(points, pose, goal);
    let states = se2_search(&w, pose, goal, emb, emb.precision)?;
    Some(densify(&states, resolution))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ring of points (square outline) at z, spacing `step`, half-size `h`.
    fn ring(cx: f64, cy: f64, h: f64, step: f64) -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        let mut t = -h;
        let z = 0.2;
        while t <= h {
            pts.push([cx - h, cy + t, z]);
            pts.push([cx + h, cy + t, z]);
            pts.push([cx + t, cy - h, z]);
            pts.push([cx + t, cy + h, z]);
            t += step;
        }
        pts
    }

    #[test]
    fn determinism() {
        let pts = ring(2.0, 0.0, 0.25, 0.05);
        let emb = Emb::go2();
        let a = plan(&pts, (0.0, 0.0, 0.0), (4.0, 0.0), &emb, 0.1).unwrap();
        let b = plan(&pts, (0.0, 0.0, 0.0), (4.0, 0.0), &emb, 0.1).unwrap();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            for k in 0..3 {
                assert_eq!(p[k].to_bits(), q[k].to_bits());
            }
        }
    }

    #[test]
    fn thin_wall_not_hopped() {
        // A tiny body blocks only ONE lattice column at the wall: without the
        // knight midpoint checks the search would hop straight through.
        let emb = Emb {
            length: 0.06,
            width: 0.06,
            center_off: 0.0,
            comfort: 0.4,
            precision: 0.01,
            strafe: 1.8,
            reverse: 1.5,
            yaw_w: 0.25,
        };
        let mut pts = Vec::new();
        let mut y = -4.0;
        while y <= 4.0 {
            pts.push([2.0, y, 0.2]);
            y += 0.02;
        }
        let path = plan(&pts, (0.0, 0.0, 0.0), (4.0, 0.0), &emb, 0.1).unwrap();
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            if (a[0] - 2.0) * (b[0] - 2.0) < 0.0 {
                let t = (2.0 - a[0]) / (b[0] - a[0]);
                let y = a[1] + t * (b[1] - a[1]);
                assert!(y.abs() > 3.8, "path hopped the wall at y={y:.2}");
            }
        }
    }

    #[test]
    fn sealed_box_refuses() {
        let pts = ring(0.0, 0.0, 1.0, 0.02);
        assert!(plan(&pts, (0.0, 0.0, 0.0), (4.0, 0.0), &Emb::go2(), 0.1).is_none());
    }
}
