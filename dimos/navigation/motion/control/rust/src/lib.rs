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

//! Onboard trajectory controller for the RK3588: the motion2 pursuit law
//! with no python in the tick.
//!
//! RULES. The law is a PORT, not a redesign -- `controller.py`'s
//! `PursuitController.update` is the specification and
//! `control/test_rust_parity.py` holds the two to 1e-9 per component. Any
//! change to the algorithm has to land on the python side first. Stateless
//! per tick (the python `reset()` is a no-op), single-threaded,
//! deterministic; dependencies stay at pyo3/numpy.
//!
//! NUMERICS. Parity is per-operation, not per-formula: `pursuit.rs` keeps
//! the python's operation ORDER and its exact tie-breaks (`argmin` takes the
//! first minimum, `searchsorted` is side='left'), and angle wrapping is IEEE
//! remainder like `math.remainder`, never `%` or `rem_euclid`.

pub mod pursuit;

#[cfg(feature = "python")]
mod python;
