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

//! Autoresearch candidate for the motion2 planning environment.
//!
//! HARD RULE — SINGLE-THREADED ONLY. No rayon, no std::thread, no crossbeam:
//! determinism is a scoring pillar (parallel float reductions are
//! order-dependent) and the deployment budget is one core on a shared RK3588
//! (the eval will run pinned via taskset, so threads would not help anyway).
//! Keep dependencies to pyo3/numpy. Rewrite the ALGORITHM in planner.rs; the
//! python-facing surface in python.rs stays stable.

pub mod planner;

#[cfg(feature = "python")]
mod python;
