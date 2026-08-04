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

//! Re-express tilted-sensor LIO odometry in the level robot body frame.
//!
//! Port of `mls_planner/odom_body_frame.py`. No tf: the mount rotation is
//! config, so this runs on a robot that has no transform tree at all.

use std::time::Duration;

use dimos_module::{error_throttled, native_config, Input, Module, Output};
use lcm_msgs::geometry_msgs::Quaternion;
use lcm_msgs::nav_msgs::Odometry;
use validator::ValidationError;

#[native_config]
#[validate(schema(function = "validate_mount_rotation"))]
pub struct Config {
    /// base_link from sensor mount rotation, xyzw. Identity says the sensor is
    /// already level, which is true of no robot that tilts its lidar: the Go2's
    /// value is `_mount_rotation()` in the go2 zenoh blueprints, and a baked
    /// host has to be handed it, since `--emit-config` emits the default.
    pub mount_rotation: [f64; 4],
    pub body_frame_id: String,
}

/// A zero quaternion has no inverse, and would otherwise turn every pose into
/// NaN one message at a time.
fn validate_mount_rotation(config: &Config) -> Result<(), ValidationError> {
    if Quat::from_xyzw(config.mount_rotation).norm_sq() == 0.0 {
        return Err(ValidationError::new(
            "mount_rotation must be a non-zero quaternion (xyzw)",
        ));
    }
    Ok(())
}

#[derive(Module)]
#[module(name = "odom_body_frame", setup = invert_mount)]
pub struct OdomBodyFrame {
    #[input(decode = Odometry::decode, handler = on_odometry)]
    odometry: Input<Odometry>,

    #[output(encode = Odometry::encode)]
    body_odometry: Output<Odometry>,

    #[config]
    config: Config,

    /// `config.mount_rotation` inverted, computed once in setup.
    mount_inv: Quat,
}

impl OdomBodyFrame {
    async fn invert_mount(&mut self) {
        self.mount_inv = Quat::from_xyzw(self.config.mount_rotation).inverse();
    }

    /// Compose the mount out of the orientation. Position, twist, both
    /// covariances, frame and stamp ride through on the message itself.
    async fn on_odometry(&mut self, mut msg: Odometry) {
        let orientation = &mut msg.pose.pose.orientation;
        *orientation = Quat::from(&*orientation).mul(self.mount_inv).into();
        msg.child_frame_id = self.config.body_frame_id.clone();

        if let Err(e) = self.body_odometry.publish(&msg).await {
            error_throttled!(
                Duration::from_secs(1),
                error = %e,
                topic = %self.body_odometry.topic,
                "Body odometry failed to publish",
            );
        }
    }
}

/// A quaternion in the dimos wire order, xyzw.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Quat {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

/// Identity, so a mount that was never inverted degrades to a passthrough
/// rather than to zeros.
impl Default for Quat {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Quat {
    fn from_xyzw([x, y, z, w]: [f64; 4]) -> Self {
        Self { x, y, z, w }
    }

    fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Hamilton product in the dimos convention: `a.mul(b)` rotates by `b`
    /// first and then by `a` (`Quaternion.__mul__`, Quaternion.py:220-235).
    /// Copied term for term from the python — the composition order is what
    /// steers the robot off-heading when it is backwards.
    fn mul(self, other: Quat) -> Quat {
        Quat {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    /// Conjugate over the squared norm (`Quaternion.inverse`,
    /// Quaternion.py:255-271). The python skips the division when the norm is
    /// within numpy's `isclose` of 1; on a unit quaternion the two agree to
    /// about an ulp, so this always divides rather than reproducing numpy's
    /// tolerances.
    fn inverse(self) -> Quat {
        let n = self.norm_sq();
        Quat {
            x: -self.x / n,
            y: -self.y / n,
            z: -self.z / n,
            w: self.w / n,
        }
    }
}

impl From<&Quaternion> for Quat {
    fn from(q: &Quaternion) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

impl From<Quat> for Quaternion {
    fn from(q: Quat) -> Self {
        Quaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Go2's real mount, `MID360_MOUNT_RPY_DEG = (-60, 0, -90)` through
    /// `Quaternion.from_euler`. Everything below was printed by the python so
    /// the two implementations are pinned to the same numbers;
    /// `test_odom_body_frame.py` asserts the python still produces them.
    const MOUNT: [f64; 4] = [
        -0.35355339059327373,
        0.3535533905932737,
        -0.6123724356957945,
        0.6123724356957946,
    ];
    /// `Quaternion.from_euler(Vector3(-0.24, 0.13, 0.7))` — a body that is
    /// rolled, pitched and yawed, so no term of the product drops out.
    const BODY: [f64; 4] = [
        -0.1343293999004263,
        0.01961508174106149,
        0.3470173839448213,
        0.9279815710081614,
    ];
    /// `BODY * MOUNT` — what the tilted sensor reports for that body pose.
    const SENSOR: [f64; 4] = [
        -0.5450515607112322,
        0.13515397172907628,
        -0.39632409041884364,
        0.7263466221066786,
    ];
    /// `SENSOR * MOUNT.inverse()` — the python's answer, which is BODY again
    /// up to float rounding.
    const LEVELED: [f64; 4] = [
        -0.13432939990042636,
        0.019615081741061635,
        0.3470173839448213,
        0.9279815710081614,
    ];

    fn assert_close(got: Quat, want: [f64; 4]) {
        let want = Quat::from_xyzw(want);
        for (g, w) in [
            (got.x, want.x),
            (got.y, want.y),
            (got.z, want.z),
            (got.w, want.w),
        ] {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn matches_the_python_on_the_go2_mount() {
        let leveled = Quat::from_xyzw(SENSOR).mul(Quat::from_xyzw(MOUNT).inverse());
        assert_close(leveled, LEVELED);
    }

    /// Get the composition order backwards and this is what fails: the
    /// round trip lands on some other rotation entirely.
    #[test]
    fn composing_the_mount_out_recovers_the_body() {
        let sensor = Quat::from_xyzw(BODY).mul(Quat::from_xyzw(MOUNT));
        assert_close(sensor, SENSOR);
        assert_close(sensor.mul(Quat::from_xyzw(MOUNT).inverse()), BODY);
    }

    #[test]
    fn identity_mount_leaves_the_orientation_alone() {
        let identity = Quat::default();
        assert_eq!(identity.inverse(), identity);
        assert_close(Quat::from_xyzw(SENSOR).mul(identity), SENSOR);
    }

    #[test]
    fn inverse_divides_by_the_squared_norm() {
        // A deliberately non-unit quaternion, where conjugate and inverse differ.
        let q = Quat {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 1.0,
        };
        assert_close(q.inverse(), [0.0, 0.0, -0.5, 0.5]);
        assert_close(q.mul(q.inverse()), [0.0, 0.0, 0.0, 1.0]);
    }

    fn config(mount_rotation: [f64; 4]) -> Config {
        Config {
            mount_rotation,
            body_frame_id: "base_link".into(),
        }
    }

    #[test]
    fn a_zero_mount_is_rejected_at_startup() {
        assert!(validate_mount_rotation(&config([0.0; 4])).is_err());
        assert!(validate_mount_rotation(&config(MOUNT)).is_ok());
    }
}
