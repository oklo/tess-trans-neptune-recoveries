# kbo-tess

Minimal Python tools for detector-local shift-stack recoveries of faint moving
objects in TESS data.

The current milestone is a bounded, known-path Sedna study for TESS Sector 5,
camera 1, CCD 4. The project has moved beyond synthetic scaffolding into a
real-data rehearsal and control-study phase, but it does not make a Sedna
recovery claim.

Current status:
- The canonical seam is a normalized ephemeris CSV plus a local NPZ cube bundle.
- The real-data moving-cutout path works through `tess_ephem` and `tess_asteroids`.
- Geometry passes on the bounded real Sedna subset tested so far.
- On that subset, `barycentric=true` is slightly better geometrically than `false`.
- The most self-consistent bounded configuration so far is:
  `barycentric=true` + loader background `rolling` + recovery background `none`.
- Injection tests recover coherent track-aligned sources strongly.
- The nominal Sedna track does not yet beat nearby null controls cleanly.
- Pluto is the next intended calibration target, but that work has not started here.

Install the baseline dependencies with:

`python3 -m pip install -e .`

For the optional real-data path, install:

`python3 -m pip install -e '.[tess]'`

That path is currently most reliable on Python 3.11-3.12 because upstream
`tess-asteroids` releases do not yet install cleanly on Python 3.13+.

Local-file workflow:
- Recovery-ready ephemeris CSV with canonical columns including
  `time_tdb_jd_minus_2457000`, `time_reference`, `x`, `y`, `sector`, `camera`,
  and `ccd`
- Detector-local NPZ cube bundle with `cube`,
  `time_tdb_jd_minus_2457000`, `local_x`, `local_y`, and `metadata_json`

Suggested bounded Sedna workflow:
- Normalize or inventory the detector track:
  `python3 scripts/sedna_inventory.py --target Sedna --sector 5 --camera 1 --ccd 4 --outdir outputs`
- Build the bounded real moving-cutout bundle:
  `python3 scripts/build_sedna_sector5_cube.py --ephem-csv outputs/sedna_sector5_inventory.csv --shape 21 21 --time-start 1438.5008 --time-stop 1440.6008 --barycentric-mode true --bg-method rolling --outdir outputs`
- Check timing and geometry before stacking:
  `python3 scripts/check_sedna_geometry.py --ephem-csv outputs/sedna_sector5_inventory.csv --cube-npz outputs/sedna_sector5_cam1_ccd4_cube_bundle.npz --outdir outputs`
- Run known-path recovery on the local bundle:
  `python3 scripts/recover_sedna_sector5.py --ephem-csv outputs/sedna_sector5_inventory.csv --cube-npz outputs/sedna_sector5_cam1_ccd4_cube_bundle.npz --background-method none --offset-grid-radius 1.0 --offset-grid-step 0.5 --interpolation linear --outdir outputs`
- Compare barycentric timing interpretations on the same subset:
  `python3 scripts/compare_barycentric_modes.py --ephem-csv outputs/sedna_sector5_inventory.csv --outdir outputs/real_control_study`
- Run injection calibration and null-track controls on a chosen bundle:
  `python3 scripts/run_injection_calibration.py --ephem-csv outputs/sedna_sector5_inventory.csv --cube-npz outputs/real_control_study/barycentric/sedna_sector5_cam1_ccd4_barycentric_cube_bundle.npz --background-method none --outdir outputs/real_control_study/injection_barycentric_none`
  `python3 scripts/run_null_track_controls.py --ephem-csv outputs/sedna_sector5_inventory.csv --cube-npz outputs/real_control_study/barycentric/sedna_sector5_cam1_ccd4_barycentric_cube_bundle.npz --background-method none --outdir outputs/real_control_study/null_controls_barycentric_none`

Generated `outputs/` are local working products and are not intended to be
committed by default. Small fixtures under `tests/fixtures/` are the preferred
way to preserve lightweight reproducible examples.
