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

//! The precision profile the planner stamps into the path's own timestamps.
//!
//! Shared facility, not a law: this is the wire dialect, and the python side
//! of it is `control/profile.py` (`encode_precision` / `decode_ceilings`),
//! which is the specification. Any law may read it; the blind track is the
//! one that has to, because it is the only channel through which required
//! precision reaches a follower that gets no clearance array.

use crate::geom::Params;

/// A segment shorter than this is a rotation in place, not a move -- the
/// python `profile._FAN_EPS`, and it has to match: the encoder prices fan
/// segments by yaw span instead of by clearance, so their dt carries no
/// precision and the decoder must skip them rather than read a speed out of
/// them.
pub const FAN_EPS: f64 = 1e-6;

/// Per-waypoint speed ceiling (m/s) recovered from the stamps, or `None` when
/// the producer does not speak the dialect.
///
/// `profile.py` states it: `ts[i] - ts[i-1] = segment length / governor speed
/// for that segment`, where the governor speed is the clearance curve
/// evaluated at the tighter of the segment's two endpoints. So the inverse is
/// division -- `ds/dt` recovers `min(gov(clr[i-1]), gov(clr[i]))` exactly, in
/// m/s, with no need to round-trip through a synthetic clearance.
///
/// A port of `profile.decode_ceilings`, deliberately statement for statement:
///
/// * Fewer than two poses, or stamps that are flat or go backwards, mean the
///   producer does not speak the dialect. Returns `None`; the caller then
///   cruises at `max_speed`.
/// * Fan segments inherit the previous ceiling.
/// * The result is clipped into `[min_speed, max_speed]`, which is what makes
///   the channel safe to trust: a stamp can only ever ask the robot to be
///   *more* careful than cruise, never faster, and garbage stamps (a path
///   whose poses were default-constructed microseconds apart, say) saturate
///   at cruise instead of commanding something absurd.
///
/// What this is NOT is a schedule. The stamps are consumed as a per-waypoint
/// speed *ceiling* keyed to arc position; the absolute times, the plan's t0
/// and the tick clock never enter. Chasing the timeline would mean
/// accelerating to make up lost time in precisely the tight passages the
/// encoding is warning about -- `profile.py` says so in as many words, and it
/// is why `t` is not marshalled into any law that reads this.
/// `ds` comes from the waypoints, NOT from differencing cumulative arc length.
/// The two are not bit-identical, the encoder used the raw segment, and the
/// python twin (`profile.decode_ceilings`) uses the raw segment too -- so
/// reconstructing it from `arcs` costs parity for nothing.
pub fn decode_ceilings(ts: &[f64], path: &[[f64; 3]], cfg: &Params) -> Option<Vec<f64>> {
    let n = path.len();
    if n < 2 || ts.len() != n {
        return None;
    }
    // `np.any(dt < 0) or not np.any(dt > 0)` -- unstamped paths (all-equal
    // ts) and anything non-monotone are rejected outright.
    let mut any_positive = false;
    for k in 1..n {
        let dt = ts[k] - ts[k - 1];
        if dt < 0.0 {
            return None;
        }
        if dt > 0.0 {
            any_positive = true;
        }
    }
    if !any_positive {
        return None;
    }
    // min/max rather than the raw fields: `f64::clamp` panics when the bounds
    // cross, and a config is free to set min_speed above max_speed.
    let lo = cfg.min_speed.min(cfg.max_speed);
    let hi = cfg.max_speed.max(cfg.min_speed);
    let mut out = vec![hi; n];
    let mut prev = hi;
    for k in 1..n {
        let (dx, dy) = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]);
        let ds = (dx * dx + dy * dy).sqrt();
        let dt = ts[k] - ts[k - 1];
        if ds >= FAN_EPS && dt > 0.0 {
            let v = ds / dt;
            // NaN would propagate straight out through the twist; the python
            // cannot produce one here because its inputs are the encoder's,
            // but this law takes whatever the wire hands it.
            if v.is_finite() {
                prev = v.clamp(lo, hi);
            }
        }
        out[k] = prev;
    }
    out[0] = out[1];
    Some(out)
}

/// The tightest decoded ceiling within `speed_lookahead` of `arcs[i]`.
///
/// Read from `i + 1` rather than `i`. That is not an off-by-one: a decoded
/// ceiling is a property of the SEGMENT ending at its waypoint, so
/// `ceilings[k]` already carries `clr[k-1]`. Scanning `[i+1 ..]` therefore
/// reproduces `gov(min clr over [i ..])` -- the clearance governor's window
/// exactly -- whereas starting at `i` would drag in the waypoint behind the
/// robot.
pub fn ceiling_ahead(ceilings: &[f64], arcs: &[f64], i: usize, cfg: &Params) -> f64 {
    let n = arcs.len();
    let hi = arcs[i] + cfg.speed_lookahead;
    // The segment about to be traversed always counts, even if a degenerate
    // `speed_lookahead` would exclude it; at the end of the plan there is no
    // next segment and the last ceiling stands.
    let mut room = ceilings[(i + 1).min(n - 1)];
    for k in (i + 1)..n {
        if arcs[k] > hi {
            break; // arcs are non-decreasing, so nothing later qualifies
        }
        if ceilings[k] < room {
            room = ceilings[k];
        }
    }
    room
}
