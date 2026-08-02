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

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::pursuit::{update as update_impl, Params};

/// One controller tick. `path` is an (N, 3) float64 array of (x, y, yaw) in
/// the pose's frame; `clearance` is an optional length-N float64 room
/// annotation (any other length is ignored, as in the python). `params`
/// carries the numeric `ControllerConfig` fields in declaration order:
/// (lookahead, max_speed, max_yaw_rate, k_pos, k_yaw, fan_yaw_per_m,
/// fan_yaw_done, min_speed, speed_clearance, speed_floor_clearance,
/// speed_lookahead). Returns the body-frame twist (vx, vy, wz).
#[pyfunction]
#[pyo3(signature = (pose, path, clearance, params))]
#[allow(clippy::type_complexity)]
fn update(
    py: Python<'_>,
    pose: (f64, f64, f64),
    path: PyReadonlyArray2<'_, f64>,
    clearance: Option<PyReadonlyArray1<'_, f64>>,
    params: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
) -> PyResult<(f64, f64, f64)> {
    if path.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "path must be (N, 3) float64, got shape {:?}",
            path.shape()
        )));
    }
    let view = path.as_array();
    let rows: Vec<[f64; 3]> = (0..view.shape()[0])
        .map(|k| [view[[k, 0]], view[[k, 1]], view[[k, 2]]])
        .collect();
    // copied rather than borrowed as a slice: a strided or non-contiguous
    // view has no slice, and the annotation is one float per waypoint
    let clr: Option<Vec<f64>> = clearance
        .as_ref()
        .map(|c| c.as_array().iter().copied().collect());
    let cfg = Params {
        lookahead: params.0,
        max_speed: params.1,
        max_yaw_rate: params.2,
        k_pos: params.3,
        k_yaw: params.4,
        fan_yaw_per_m: params.5,
        fan_yaw_done: params.6,
        min_speed: params.7,
        speed_clearance: params.8,
        speed_floor_clearance: params.9,
        speed_lookahead: params.10,
    };
    Ok(py.allow_threads(|| update_impl(pose, &rows, clr.as_deref(), &cfg)))
}

#[pymodule]
fn dimos_motion2_tc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update, m)?)?;
    Ok(())
}
