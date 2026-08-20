# Trial context — PR #3162 (fiducial relocalization prior)

Everything written during the trial that PR #3162 still needs, kept verbatim. This branch is a
sibling of `feat/relocalization-fiducial-prior` and is not part of the PR — it exists so the PR
diff stays small while the reasoning behind it survives.

## Read first, to finish the PR

| File | What it holds |
|---|---|
| `FINDINGS.md` | Every measured result + the honesty log of three claims that turned out wrong. |
| `PR_fiducial_relocalization.md` | PR body draft, house template. |
| `PR_priors_config.md` | Second PR body draft, per-prior config framing. |
| `PLAN_prior_config.md` | Design for the follow-up `priors: list[PriorConfig]` refactor. Ends in 4 open questions Aaryan never answered. |
| `PLAN.md` | The original goal + architecture, with build checkboxes. |
| `pr3162_simplify_verified.patch` | Verified simplification pass, unapplied. |

## The rest

| File | What it holds |
|---|---|
| `harness/BENCHMARK_METHOD.md` | The grade-real-dimos rule and the body-frame scar that produced it. |
| `harness/PROVENANCE.md` | 108 constants audited: 49 arbitrary, 30 partial, 29 tested. |
| `harness/README.md` | How the offline benchmark ran. |
| `harness/benchmark_setup.yaml` | Recording manifest — the standing marker-truth setup. |
| `office_markers.yaml` | sf_office marker survey. |
| `DEMO_leshy.md` | Replay demo runbook on hk_village3, three blueprints. |
| `quiz.md` | Question bank, answers cited to file:line. |
| `PR_camera_calibration.md` | Adjacent PR draft: charuco + `--check`. |
| `robotday_kit/` | Printable fiducial and referee tag sheets, as PDFs. The four PNG previews are not here: `.gitattributes` routes `*.png` through LFS, and a new LFS object cannot be pushed from a box without `lfs.dimensionalos.com` credentials. Regenerate any sheet with `dimos apriltag`. |

## What is not here

The harness source (24 Python files) and 27 result figures were deleted from the
`aaryan-dimensional` repo. Both are recoverable:

    git checkout c20f601 -- trial            # in aaryan-dimensional
    ls aaryan-dimensional/workspace/temp/trial   # same files, still on the dev box

Figures also live at `aaryan-dimensional/site/assets/`. The 11 GB of eval output
(`trial/harness/out`) and the `.rrd` recordings were never in git; they are in
`workspace/temp/trial` on the dev box only.

## Where the work stopped

The prior detects, gates, fuses, and proposes in 24 of 28 held-out cycles, and wins 0 — it loses to
RANSAC on wall fitness by ~0.05-0.15, not on plumbing. The gap is marker-pose accuracy: IPPE
mirror-flip on the Go2's wide fisheye plus a static, not-per-unit camera calibration (DIM-1308).
