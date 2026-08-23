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

use dimos_module::{native_config, run_with_transport, Input, Module, Output};
use lcm_msgs::geometry_msgs::Quaternion;
use lcm_msgs::sensor_msgs::Imu;
use lcm_msgs::std_msgs::Bool;
use tracing::{error, warn};

#[native_config]
struct EStopConfig {
    // Above 90 the body is already past horizontal, and 180 could never trip.
    #[validate(range(min = 1.0, max = 90.0))]
    max_tilt_deg: f64,
}

/// Latched e-stop: any check may set it, nothing in-process clears it.
#[derive(Default)]
struct Latch(bool);

impl Latch {
    /// True only on the transition into the latched state, so the signal publishes once.
    fn set(&mut self, tripped: bool) -> bool {
        let fired = tripped && !self.0;
        self.0 |= tripped;
        fired
    }
}

/// Angle between the body z axis and gravity, off R[2][2] of the quaternion, so yaw invariant.
/// None when the sample is not a usable unit quaternion, which must never read as upright.
fn tilt_deg(q: &Quaternion) -> Option<f64> {
    let norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if !norm_sq.is_finite() || (norm_sq - 1.0).abs() > 1e-3 {
        return None;
    }
    Some(
        (1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees(),
    )
}

/// A sample is fallen when it tilts past the limit, or when it cannot be read at all.
fn is_fallen(q: &Quaternion, max_tilt_deg: f64) -> bool {
    tilt_deg(q).is_none_or(|deg| deg > max_tilt_deg)
}

#[derive(Module)]
struct EStop {
    #[input(decode = Imu::decode)]
    imu: Input<Imu>,

    #[input(decode = Bool::decode)]
    trigger: Input<Bool>,

    #[output(encode = Bool::encode)]
    estop: Output<Bool>,

    #[config]
    config: EStopConfig,

    latch: Latch,
}

impl EStop {
    async fn check(&mut self, tripped: bool, reason: &'static str) {
        if self.latch.set(tripped) {
            warn!(reason, "e-stop latched");
        }
        if self.latch.0 {
            // Re-assert every sample: the transport is best effort, so one publish can be lost.
            if let Err(e) = self.estop.publish(&Bool { data: true }).await {
                error!(error = %e, "e-stop publish failed");
            }
        }
    }

    async fn handle_imu(&mut self, msg: Imu) {
        let fallen = is_fallen(&msg.orientation, self.config.max_tilt_deg);
        self.check(fallen, "tilt").await;
    }

    async fn handle_trigger(&mut self, msg: Bool) {
        self.check(msg.data, "button").await;
    }
}

#[tokio::main]
async fn main() {
    run_with_transport::<EStop>().await;
}

#[cfg(test)]
mod tests {
    use super::{is_fallen, tilt_deg, Latch, Quaternion};

    const MAX_TILT_DEG: f64 = 45.0; // the default EStopConfig ships, so the replay grades production
    const FIXTURE: &str = include_str!("../../g1_fall_imu.csv");

    fn quat(w: f64, x: f64, y: f64, z: f64) -> Quaternion {
        Quaternion { x, y, z, w }
    }

    /// Rows of the measured fall as (t_s, orientation); only data lines start with a digit.
    fn fixture_rows() -> Vec<(f64, Quaternion)> {
        FIXTURE
            .lines()
            .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
            .map(|l| {
                let v: Vec<f64> = l.split(',').map(|f| f.parse().unwrap()).collect();
                (v[0], quat(v[1], v[2], v[3], v[4]))
            })
            .collect()
    }

    /// Tilt measures the body z axis against gravity, so it is zero upright and unmoved by yaw.
    #[test]
    fn upright_reads_zero_at_any_yaw() {
        assert_eq!(tilt_deg(&quat(1.0, 0.0, 0.0, 0.0)).unwrap(), 0.0);
        for yaw_rad in [0.5f64, 2.0, 3.0] {
            let (s, c) = (yaw_rad / 2.0).sin_cos();
            assert_eq!(tilt_deg(&quat(c, 0.0, 0.0, s)).unwrap(), 0.0);
        }
    }

    /// A 90 deg roll lays the body z axis into the horizontal plane, 90 deg off gravity.
    #[test]
    fn ninety_degree_roll_reads_ninety() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((tilt_deg(&quat(h, h, 0.0, 0.0)).unwrap() - 90.0).abs() < 1e-9);
    }

    /// An unusable orientation returns None so it can never be mistaken for upright.
    #[test]
    fn invalid_orientation_is_not_upright() {
        assert_eq!(tilt_deg(&quat(f64::NAN, 0.0, 0.0, 0.0)), None);
        assert_eq!(tilt_deg(&quat(0.0, 0.0, 0.0, 0.0)), None);
        let s = (60f64.to_radians() / 2.0).sin();
        let c = (60f64.to_radians() / 2.0).cos();
        assert_eq!(tilt_deg(&quat(0.7 * c, 0.7 * s, 0.0, 0.0)), None);
    }

    /// A broken IMU must stop the robot: an unreadable sample counts as fallen, not as upright.
    #[test]
    fn unreadable_imu_counts_as_fallen() {
        assert!(is_fallen(&quat(f64::NAN, 0.0, 0.0, 0.0), MAX_TILT_DEG));
        assert!(is_fallen(&quat(0.0, 0.0, 0.0, 0.0), MAX_TILT_DEG));
        assert!(!is_fallen(&quat(1.0, 0.0, 0.0, 0.0), MAX_TILT_DEG));
    }

    /// The manual button latches on its first true, and no later message clears it.
    #[test]
    fn manual_trigger_latches() {
        let mut latch = Latch::default();
        assert!(latch.set(true));
        assert!(!latch.set(true));
        assert!(!latch.set(false));
        assert!(latch.0);
    }

    /// Replaying the measured fall trips once and holds through the recovery between the two falls.
    #[test]
    fn measured_fall_trips_once_and_holds() {
        let rows = fixture_rows();
        assert_eq!(rows.len(), 1401);

        let mut latch = Latch::default();
        let fired: Vec<f64> = rows
            .iter()
            .filter(|(_, q)| latch.set(tilt_deg(q).unwrap() > MAX_TILT_DEG))
            .map(|(t_s, _)| *t_s)
            .collect();
        assert_eq!(fired, vec![8.37]); // 45.90 deg, the first sample over the threshold
        assert!(latch.0);

        let recoveries = rows
            .windows(2)
            .filter(|w| tilt_deg(&w[0].1).unwrap() > MAX_TILT_DEG && tilt_deg(&w[1].1).unwrap() <= MAX_TILT_DEG)
            .count();
        assert_eq!(recoveries, 1); // the run the latch has to survive; without it this test is vacuous
    }
}
