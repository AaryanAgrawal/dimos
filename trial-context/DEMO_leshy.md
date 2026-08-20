# Demo runbook — relocalization priors (lidar / lidar+fiducial / fiducial-only)

For leshy, 2026-07-22. Replay demo on `hk_village3`. Everything below was run end to end on this box at
dimos `cdbacdd16` (branch `feat/relocalization-fiducial-prior`); HEAD is now `3471c8cae`, same branch.
PR: https://github.com/dimensionalOS/dimos/pull/3137 (open, head `cdbacdd16`; the two later commits are
local). Talking points for the call are §6.

---

## 1. What to show, in order

| # | Blueprint | What it demonstrates |
|---|---|---|
| 1 | `unitree-go2-relocalization-lidar` | Baseline: RANSAC global registration alone publishes `world -> map`. |
| 2 | `unitree-go2-relocalization-lidar-fiducial` | Both priors composed: fiducial proposes a candidate every cycle, RANSAC proposes ~34, the judge picks per-cycle on fitness. |
| 3 | `unitree-go2-relocalization-fiducial` | Fiducial alone, RANSAC removed: relocalizes from one AprilTag in ~0.2 s vs ~4.4 s, and shows the fitness health signal collapsing when the single-tag prior goes bad. |

Run 2 is the one to spend time on. Run 3 is the interesting one, but read §4 before demoing it.

---

## 2. Commands

All commands from `/home/dimos/dimensional-trial/dimos`. One at a time — the LCM bus is exclusive.

Shorthand:

```
PREMAP=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.pc2.lcm
MARKERS=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.markers.yaml
```

### Replay (premap + marker map given)

```bash
# 1. lidar only — no marker map needed
cd /home/dimos/dimensional-trial/dimos && uv run dimos --replay --replay-db=hk_village3 \
  run unitree-go2-relocalization-lidar \
  -o relocalizationmodule.map_file=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.pc2.lcm \
  -o relocalizationmodule.publish_loaded_map=true
```

```bash
# 2. lidar + fiducial
cd /home/dimos/dimensional-trial/dimos && uv run dimos --replay --replay-db=hk_village3 \
  run unitree-go2-relocalization-lidar-fiducial \
  -o relocalizationmodule.map_file=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.pc2.lcm \
  -o relocalizationmodule.marker_map_file=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.markers.yaml \
  -o relocalizationmodule.publish_loaded_map=true
```

```bash
# 3. fiducial only
cd /home/dimos/dimensional-trial/dimos && uv run dimos --replay --replay-db=hk_village3 \
  run unitree-go2-relocalization-fiducial \
  -o relocalizationmodule.map_file=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.pc2.lcm \
  -o relocalizationmodule.marker_map_file=/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.markers.yaml \
  -o relocalizationmodule.publish_loaded_map=true
```

Each replay covers the recording in ~110 s after module start. Stop with Ctrl-C.

### Live robot (same three, if a Go2 is on the network)

Swap `--replay --replay-db=hk_village3` for `--robot-ip <ip>` and point at a premap + marker map
surveyed in that room. Last verified robot IP was `10.0.0.104` — re-sweep before the call, the printed
label on the robot (`192.168.10.190`) is wrong. Firmware >= 1.1.15 also needs
`--unitree-aes-128-key <key>`.

```bash
cd /home/dimos/dimensional-trial/dimos && uv run dimos --robot-ip 10.0.0.104 \
  run unitree-go2-relocalization-lidar-fiducial \
  -o relocalizationmodule.map_file=<abs path>.pc2.lcm \
  -o relocalizationmodule.marker_map_file=<abs path>.markers.yaml \
  -o relocalizationmodule.publish_loaded_map=true
```

The tag size and dictionary are baked into the blueprint, not operator-settable: **0.10 m edge,
`DICT_APRILTAG_36h11`**. Live tags that are a different size need a code change, not a flag.

### Viewer on another machine

Add `--rerun-open none` before `run`. The bridge then prints the connect line:
`dimos-viewer --connect rerun+http://<host>:9877/proxy --ws-url ws://<host>:3030/ws`.
`--rerun-web` serves the web viewer instead. Default (`native`) spawns a local viewer window.

---

## 3. Where to look

**Rerun entity paths** (rerun is on by default in all three blueprints, nothing to add):

| Path | What |
|---|---|
| `world/tf/map` | **The fix.** The `world -> map` transform, redrawn every 2 s once accepted. |
| `world/merged_map` | Premap transformed by the fix + live map, carved. Visual proof the fix is right — the premap should snap onto the live cloud. |
| `world/global_map` | Live voxel map, the reloc input. |
| `world/loaded_map` | Raw premap (only with `publish_loaded_map=true`). |
| `world/color_image` | Camera. Tag 10 is visible here on the fiducial runs. |

**LCM topics:**

| Topic | Type | Note |
|---|---|---|
| `/tf` | `TFMessage` | **The world->map fix msgs.** `frame_id="world"`, `child_frame_id="map"`. Published by `RelocalizationModule._publish_periodic`, every 2.0 s after acceptance. |
| `/merged_map` | `PointCloud2` | premap ∘ fix + carved live map |
| `/loaded_map` | `PointCloud2` | raw premap, gated on `publish_loaded_map` |
| `/global_map` | `PointCloud2` | voxel mapper output |
| `/detections` | `Detection3DArray` | marker sightings, fiducial blueprints only |

**Log lines to read out loud:**

```
fiducial prior enabled marker_map_file=... n_markers=1
relocalize accepted fitness=0.984 n_pts=51034 published_t_m=[...] source=ransac tf_from=world tf_to=map time_cost_s=1.7
relocalize rejected / relocalize skipped
```

`source=` names the winning prior. It is absent on the `-lidar` run — single-source path, by design.

---

## 4. What works / what doesn't

Measured, not estimated. Full logs: `/home/dimos/dimensional-trial/trial/harness/out/hk_village3_bp_runs/run{1,2,3}_*.log`
(first line of each is the exact command).

| | `-lidar` | `-lidar-fiducial` | `-fiducial` |
|---|---|---|---|
| accepted | 17 | 20 | 21 |
| rejected | 0 | 0 | 0 |
| skipped (warmup) | 4 | 4 | 4 |
| fitness range (median) | 0.798–0.984 (0.890) | 0.796–0.984 (0.941) | 0.590–0.984 (0.865) |
| winning `source=` | n/a | ransac 20/20, fiducial **0**/20 | fiducial 21/21 |
| candidates per cycle | n/a | `{fiducial: 1, ransac: 34}` | `{fiducial: 1}` |
| `time_cost_s` median | 4.4 | 4.0 | **0.2** |

**Works**

- All three blueprints compose, start, and publish `world -> map`. Zero rejects across 58 fixes.
- The fiducial prior loads and engages: `n_markers=1`, one candidate proposed on **every** cycle in both
  fiducial blueprints. The wiring is real, not stubbed.
- Fiducial-only relocalizes from a single tag at ~20x the speed of RANSAC (0.2 s vs 4.4 s) because it
  skips global registration entirely.
- Fusion arbitration is correct: in `-lidar-fiducial` the judge picked RANSAC on all 20 cycles,
  including the exact window where fiducial-only went ~1 m wrong. Fitness never dipped below 0.796.

**Doesn't**

- **The fiducial prior contributes zero winning fixes on `hk_village3`.** It proposes, it loses, every
  cycle. It costs nothing and it adds nothing here. Do not claim it improved accuracy on this recording.
- **Cause on this recording is a weak map, not a flip.** `hk_village3` has exactly **one** distinct tag
  id (10). Its 4 detection tracks spread up to **1.38 m** apart, so the medoid pose is robust but not
  precise. One tag with metre-scale inter-track disagreement cannot outbid a 34-candidate RANSAC field.
  There is **no mirror flip in this recording** — checked; the ambiguity gate is on (`ambiguity_ratio_min=2.0`)
  and gate-on vs gate-off marker maps differ by only 0.023 m / 2.19°.
- **`sf_office` is a different recording with a different result — don't merge the stories.** Held out
  (survey1 premap → survey2 replay, shipped defaults): 32 cycles, 21 accepted, winners ransac 18 at
  median fitness 0.661 and **fiducial 3 at median 0.687**. Marker-pose quality there is still the known
  weakness (Go2 camera calibration + an IPPE mirror flip baked into the survey, DIM-1308). Any earlier
  "fiducial won 0/28 on sf_office" line is dead: it came from a harness parser that matched `source=`
  before `fitness=` while the module emits `source=` last, so it could only ever print `ransac`. It had
  zero measurement power. The hk_village3 counts above are read from the raw run logs, not that parser.
- **The top-10 pre-ICP cut is what actually eliminates the prior.** `relocalize.py` refines only the
  top-10 candidates ranked on **pre-ICP** wall fitness. Fiducial candidates are raw marker-derived poses
  with no ICP polish, so against ~34 RANSAC candidates they mostly die before the judge: on sf_office
  proposed in 27/32 cycles, reached the judge 5 times, won 3 of those 5. That is the architectural
  question for the call (§6b), not a tuning issue.
- **Fiducial-only degrades in the last third and nothing catches it.** Paired against lidar-only at
  matched replay progress, `world_T_map` translation, meters:

  ```
    n_pts  fit_fid  fit_lid  |dxy| m
    52952    0.984    0.984    0.004   <- first 14 fixes track lidar to 0.001-0.25 m
   100278    0.833    0.804    0.029
   100691    0.591    0.804    0.843   <- regime change, one cycle
   104079    0.602    0.830    0.963
    96760    0.603    0.806    1.372   <- worst
  ```

  The fitness collapse (0.83 -> 0.59 in one cycle) is a genuine health signal that fires exactly when the
  fix goes wrong — but `fitness_threshold=0.45`, so all seven ~1 m-wrong fixes are accepted and published.
  Max divergence 1.37 m matches the pre-flagged 1.38 m track spread. Expected, not a new bug.
  If demoing run 3: either stop before the 100k-point mark, or show the collapse deliberately as the
  health signal doing its job.

**Not verified:** fixes were confirmed via the `relocalize accepted ... tf_from=world tf_to=map` log and
the downstream planner consuming `/merged_map`, not by sniffing LCM `/tf` directly.

---

## 5. Prereqs and gotchas

**Files that must exist** (all present on this box):

```
/home/dimos/dimensional-trial/dimos/data/hk_village3.db                        325 MB, replay source
/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.pc2.lcm       premap, 267,849 pts, frame_id world
/home/dimos/dimensional-trial/dimos/data/replay_gate/hk_village3.markers.yaml  marker map, 1 id: 10
```

Marker map contents — `map_T_marker_10`, PGO-corrected map frame, meters + quaternion xyzw:

```
translation: [-4.168194109163993, -30.983437065609696,  0.26734221161092914]
rotation:    [ 0.1984921085003767, -0.6801881381906297, -0.6822754116070265, 0.1801256290601145]
```

Sanity: 0.013 m from the nearest premap point, 12,735 premap points within 1 m — the pose sits on real
map geometry. Regenerate with:

```bash
cd /home/dimos/dimensional-trial/dimos && uv run python \
  /home/dimos/dimensional-trial/trial/harness/derive_marker_map.py \
  --recording hk_village3 --ambiguity-ratio-min 2.0 --out /tmp/hk_village3.markers.gate_on.yaml
```

**Gotchas**

1. **The LCM bus is exclusive.** One run at a time. Ctrl-C and wait for the process to exit before
   starting the next, or the second run silently sees the first one's traffic.
2. **Startup can die before the blueprint builds.** On this box:
   `CalledProcessError: ['sudo','ip','link','set','lo','multicast','on']` (`system_configurator/base.py:110`).
   `lo` has no MULTICAST flag and sudo is password-gated. Either run in an interactive terminal that can
   answer the sudo prompt, enable loopback multicast on the host beforehand, or prefix `PYTEST_VERSION=1`
   (short-circuits `configure_system`; host-config skip only, the pipeline stays untouched real dimos —
   that is what the recorded runs used).
3. **`uv run dimos run <go2 blueprint> --help` crashes** (`ValueError: The default factory requires the
   'validated_data' argument`, `robot/cli/dimos.py:217`). Do not discover args via `--help` on the call.
4. **Forgetting `marker_map_file` fails quietly.** On `-lidar-fiducial` you get a warning and a silent
   degrade to lidar-only. On `-fiducial` you get **zero fixes and no error**. Check for the
   `fiducial prior enabled ... n_markers=1` line in the first seconds of every fiducial run.
5. **`min_local_points=50000`** — expect ~4 `relocalize skipped` warnings before the first fix. Normal.
6. **Live runs need `--robot-ip`**; there is no `.env` on this box so `robot_ip` defaults to `None` and
   the webrtc path asserts.
7. `trial/harness/eval.py` and `record_reloc.py` still pass the removed
   `-o relocalizationmodule.use_fiducial_prior=true` and will fail at launch. Don't copy commands out of
   them, or out of `trial/harness/out/robotday_live/*/replay_cmd.txt` (dead blueprint names).

---

## 6. Talking points, ranked

**a. Fiducial relocalizes standalone, and it is ~20x faster.** `-fiducial` publishes 21/21 fixes off a
single AprilTag, median `time_cost_s` 0.2 vs 4.4 for RANSAC, because it skips global registration. On
a robot waking up in a known room that is the difference between a fix now and a fix in five seconds.

**b. The top-10 pre-ICP cut eliminates the prior before the judge sees it — the real architectural
question.** `relocalize.py` polishes only the top-10 candidates ranked on pre-ICP wall fitness. A
fiducial candidate is a raw marker-derived pose, unpolished, competing against ~34 RANSAC candidates on
exactly the metric it has not been optimised for: sf_office proposed 27/32, reached the judge 5, won 3
of those 5. So the prior is mostly cut by a ranking heuristic, not by the judge. Options: reserved slots
for prior-proposed candidates, or ICP-polish priors before ranking. Want lesh's call.

**c. Where the prior does reach the judge, it wins on merit.** sf_office held out (survey1 premap →
survey2 replay, shipped defaults, nothing tuned): 21 accepted, ransac 18 at median fitness 0.661,
fiducial 3 at median **0.687**. Small n, but it is a source-blind judge picking the marker candidate on
wall geometry.

**d. The health signal is real and wired to nothing.** On `-fiducial`, fitness collapses 0.83 → 0.59 in
one cycle exactly when the fix goes ~1 m wrong — the signal fires correctly. But `fitness_threshold` is
0.45, so all seven wrong fixes publish anyway. A relative-drop / trend gate, or a per-source threshold,
would catch it. Today nothing consumes the collapse. Worth ten minutes on the call.

**e. The "fiducial won 0/28" number was never measured.** Harness parser matched `source=` before
`fitness=`; the module emits `source=` last, so the parser could only ever emit `ransac`. Fixed
(`27d50ac72`), all counts re-derived from raw run logs. Flagging it because it shaped the earlier
"the prior never wins" framing.

**Weak-prior caveat to say out loud on run 2/3:** `hk_village3` has one tag id (10) with 1.38 m
inter-track spread. One imprecise tag cannot outbid a 34-candidate RANSAC field, and the 1.37 m worst
divergence on `-fiducial` matches that spread. It is a weak prior, not a broken one.

**Asks:** (1) reserved slots or pre-rank polish for prior candidates — b; (2) is a relative-drop
fitness gate the right home for the collapse signal — d; (3) hardware run and a clean multi-tag
recording are the two open gaps before this is more than replay evidence.
