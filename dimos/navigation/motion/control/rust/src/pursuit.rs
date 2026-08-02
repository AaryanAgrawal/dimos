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

//! Port of `control/controller.py::PursuitController.update`: pose + (x, y,
//! yaw) path in, body-frame twist out. Holonomic -- the Go2 crabs, so
//! position error maps straight to (vx, vy) and yaw tracks the path's own
//! yaw, with fan segments (the planner's commanded rotations) becoming
//! rotate-in-place.
//!
//! Pure: plain slices in, `(vx, vy, wz)` out, no pyo3 types and no state
//! between ticks. Every step below mirrors one statement of the python; the
//! comments name the python construct being reproduced wherever the Rust
//! spelling hides it.

/// The numeric half of `ControllerConfig`. `frame_id` stays python-side: it
/// selects the tf lookup, it does not enter the law.
pub struct Params {
    pub lookahead: f64,
    pub max_speed: f64,
    pub max_yaw_rate: f64,
    pub k_pos: f64,
    pub k_yaw: f64,
    pub fan_yaw_per_m: f64,
    pub fan_yaw_done: f64,
    pub min_speed: f64,
    pub speed_clearance: f64,
    pub speed_floor_clearance: f64,
    pub speed_lookahead: f64,
}

impl Default for Params {
    /// The `ControllerConfig` field defaults, verbatim.
    fn default() -> Self {
        Self {
            lookahead: 0.35,
            max_speed: 0.5,
            max_yaw_rate: 1.4,
            k_pos: 2.0,
            k_yaw: 2.0,
            fan_yaw_per_m: 3.0,
            fan_yaw_done: 0.25,
            min_speed: 0.2,
            speed_clearance: 0.35,
            speed_floor_clearance: 0.05,
            speed_lookahead: 2.0,
        }
    }
}

pub const TAU: f64 = std::f64::consts::TAU;

/// IEEE-754 remainder, i.e. `math.remainder`: the quotient rounds half to
/// EVEN, which is what puts the result in [-y/2, y/2] and what makes
/// `remainder(pi, tau)` come out `+pi` rather than `-pi`. Neither `%`
/// (truncated) nor `rem_euclid` (always non-negative) is this function.
#[inline]
pub fn ieee_remainder(x: f64, y: f64) -> f64 {
    x - (x / y).round_ties_even() * y
}

/// Shortest signed angle from `b` to `a` -- the python `_angle_diff`.
#[inline]
fn angle_diff(a: f64, b: f64) -> f64 {
    ieee_remainder(a - b, TAU)
}

/// One controller tick. `path` is the plan as (x, y, yaw) rows; `clearance`
/// is the optional per-waypoint room annotation. Returns `(vx, vy, wz)` in
/// the body frame.
pub fn update(
    pose: (f64, f64, f64),
    path: &[[f64; 3]],
    clearance: Option<&[f64]>,
    cfg: &Params,
) -> (f64, f64, f64) {
    if path.len() < 2 {
        // empty path or a single-pose veto stub: there is nothing to
        // follow -- hold position (the planner is saying "stop")
        return (0.0, 0.0, 0.0);
    }
    let (px, py, pyaw) = pose;
    let n = path.len();

    // arcs = concatenate([[0.0], cumsum(norm(diff(xy, axis=0), axis=1))];
    // cumsum accumulates left to right, so the running sum is sequential.
    let mut arcs = vec![0.0f64; n];
    for k in 1..n {
        let (dx, dy) = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]);
        arcs[k] = arcs[k - 1] + (dx * dx + dy * dy).sqrt();
    }

    // closest waypoint = progress along the path; inside a fan the
    // waypoints are coincident, so advance by yaw progress instead of
    // re-rotating from the fan's first pose
    let mut i = 0usize;
    let mut best = f64::INFINITY;
    for (k, p) in path.iter().enumerate() {
        // strict `<`: np.argmin keeps the FIRST minimum on a tie
        let d = ((p[0] - px) * (p[0] - px) + (p[1] - py) * (p[1] - py)).sqrt();
        if d < best {
            best = d;
            i = k;
        }
    }
    while i + 1 < n
        && arcs[i + 1] - arcs[i] < 1e-6
        && angle_diff(path[i + 1][2], pyaw).abs() < angle_diff(path[i][2], pyaw).abs()
    {
        i += 1;
    }

    // fan detection at the current position: yaw stepping with (near-)zero
    // displacement means the planner commands a rotation here
    let j = (i + 1).min(n - 1);
    let ds = arcs[j] - arcs[i];
    let dyaw = angle_diff(path[j][2], path[i][2]).abs();
    let in_fan = j > i && dyaw > 1e-6 && dyaw / ds.max(1e-6) > cfg.fan_yaw_per_m;
    let (target_xy, target_yaw) = if in_fan && angle_diff(path[j][2], pyaw).abs() > cfg.fan_yaw_done
    {
        ([path[i][0], path[i][1]], path[j][2])
    } else {
        let s = arcs[i] + cfg.lookahead;
        // np.searchsorted(arcs, s) with the default side='left': the first
        // index whose arc is >= s, which on sorted data is the count of
        // strictly smaller entries.
        let k = arcs.partition_point(|&a| a < s).min(n - 1);
        ([path[k][0], path[k][1]], path[k][2])
    };

    // speed governor: cap cruise by the room ahead, when we know it
    let mut vmax = cfg.max_speed;
    if let Some(clr) = clearance {
        if clr.len() == n {
            // the mask (arcs >= arcs[i]) & (arcs <= arcs[i] + lookahead) is
            // not a contiguous slice -- coincident fan waypoints before `i`
            // share its arc -- so scan the whole array like numpy does
            let hi = arcs[i] + cfg.speed_lookahead;
            let mut room: Option<f64> = None;
            for (k, &a) in arcs.iter().enumerate() {
                if a >= arcs[i] && a <= hi && room.is_none_or(|m| clr[k] < m) {
                    room = Some(clr[k]);
                }
            }
            // arcs[i] always passes its own mask, so the window is never
            // empty; the fallback is here because the python spells it out
            let room = room.unwrap_or(clr[i]);
            let frac = (room - cfg.speed_floor_clearance)
                / (cfg.speed_clearance - cfg.speed_floor_clearance).max(1e-6);
            vmax = cfg.min_speed + (cfg.max_speed - cfg.min_speed) * frac.clamp(0.0, 1.0);
        }
    }

    // body-frame error -> velocity
    let (ex, ey) = (target_xy[0] - px, target_xy[1] - py);
    let (c, s_) = ((-pyaw).cos(), (-pyaw).sin());
    let (bx, by) = (c * ex - s_ * ey, s_ * ex + c * ey);
    let (mut vx, mut vy) = (cfg.k_pos * bx, cfg.k_pos * by);
    let speed = vx.hypot(vy);
    if speed > vmax {
        vx = vx / speed * vmax;
        vy = vy / speed * vmax;
    }
    // np.clip = minimum(maximum(v, lo), hi); spelled out rather than
    // `clamp`, which panics when a config sets a negative max_yaw_rate
    let wz = (cfg.k_yaw * angle_diff(target_yaw, pyaw))
        .max(-cfg.max_yaw_rate)
        .min(cfg.max_yaw_rate);
    (vx, vy, wz)
}
