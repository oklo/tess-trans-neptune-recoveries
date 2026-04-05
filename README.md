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
- The CLIs are target-aware enough to run the same bounded workflow on Sedna and Pluto via `--target`.
- Geometry passes on the bounded real Sedna subset tested so far.
- On that subset, `barycentric=true` is slightly better geometrically than `false`.
- The most self-consistent bounded configuration so far is:
  `barycentric=true` + loader background `rolling` + recovery background `none`.
- Injection tests recover coherent track-aligned sources strongly.
- The nominal Sedna track does not yet beat nearby null controls cleanly.
- The first bounded Pluto calibration run now exists on Sector 92 / camera 2 / CCD 1:
  geometry passes and direct moving-aperture photometry is sensible, but the exact
  nominal shift-stack track still does not beat null controls cleanly.

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

Bounded Pluto calibration workflow:
- Inventory all detector-local Pluto opportunities and pick a bounded primary/backup window:
  `python3 scripts/sedna_inventory.py --target Pluto --all-detectors --window-days 4.0 --outdir outputs/pluto_calibration`
- Build the selected moving-cutout bundle:
  `python3 scripts/build_sedna_sector5_cube.py --target Pluto --sector 92 --camera 2 --ccd 1 --ephem-csv outputs/pluto_calibration/pluto_all_detector_inventory.csv --shape 21 21 --time-start 3826.5008007516153 --time-stop 3828.5008007511497 --barycentric-mode true --bg-method rolling --outdir outputs/pluto_calibration/primary`
- Check geometry and direct moving-aperture photometry:
  `python3 scripts/check_sedna_geometry.py --target Pluto --sector 92 --camera 2 --ccd 1 --ephem-csv outputs/pluto_calibration/pluto_all_detector_inventory.csv --cube-npz outputs/pluto_calibration/primary/pluto_sector92_cam2_ccd1_cube_bundle.npz --outdir outputs/pluto_calibration/primary`
  `python3 scripts/run_moving_aperture_photometry.py --target Pluto --sector 92 --camera 2 --ccd 1 --ephem-csv outputs/pluto_calibration/pluto_all_detector_inventory.csv --cube-npz outputs/pluto_calibration/primary/pluto_sector92_cam2_ccd1_cube_bundle.npz --background-method none --outdir outputs/pluto_calibration/primary`
- Run known-path recovery and null controls:
  `python3 scripts/recover_sedna_sector5.py --target Pluto --sector 92 --camera 2 --ccd 1 --ephem-csv outputs/pluto_calibration/pluto_all_detector_inventory.csv --cube-npz outputs/pluto_calibration/primary/pluto_sector92_cam2_ccd1_cube_bundle.npz --background-method none --offset-grid-radius 1.0 --offset-grid-step 0.5 --interpolation linear --outdir outputs/pluto_calibration/primary/recovery_none`
  `python3 scripts/run_null_track_controls.py --target Pluto --sector 92 --camera 2 --ccd 1 --ephem-csv outputs/pluto_calibration/pluto_all_detector_inventory.csv --cube-npz outputs/pluto_calibration/primary/pluto_sector92_cam2_ccd1_cube_bundle.npz --background-method savgol --interpolation linear --offset-grid-radius 1.0 --offset-grid-step 0.5 --outdir outputs/pluto_calibration/primary/null_controls_savgol`

Generated `outputs/` are local working products and are not intended to be
committed by default. Small fixtures under `tests/fixtures/` are the preferred
way to preserve lightweight reproducible examples.
