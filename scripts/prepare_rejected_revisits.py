"""Restore all sector footprints from immutable task bundles; acquire no images."""
import argparse
import json
from pathlib import Path
import tempfile

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from p9search.storage import DiskBudget, read_json, write_json
from p9search.tasks import fetch_bundle, validate_task
from publish_travel_reviews import revisit_predecessor


def footprint(directory, sector):
    meta = read_json(directory / "metadata.json")
    wcs = WCS(fits.Header.fromstring((directory / "wcs.txt").read_text(), sep="\n"))
    edge = np.linspace(-.5, meta["size"]-.5, 17)
    xy = np.vstack([np.column_stack([edge, np.full(17, -.5)]),
                    np.column_stack([np.full(17, meta["size"]-.5), edge]),
                    np.column_stack([edge[::-1], np.full(17, meta["size"]-.5)]),
                    np.column_stack([np.full(17, -.5), edge[::-1]])])
    sky = wcs.all_pix2world(xy + [meta["origin_column"], meta["origin_row"]], 1)
    if not np.isfinite(sky).all() or np.ptp(np.unwrap(sky[:, 0], period=360)) >= 3:
        raise ValueError("Invalid sector footprint")
    return dict(sector=sector, ra_dec_deg=sky.round(6).tolist())


def prepare(inventory, previous, destination):
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    ready, failures = [], []
    for row in read_json(inventory):
        if not row["result"]:
            continue
        task = validate_task(json.loads(row["task"]["spec"]))
        target = destination / (task["task_id"] + ".json")
        revisit_predecessor(previous, task, row)
        try:
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=destination) as temporary:
                    data = fetch_bundle(task, temporary, DiskBudget(destination, 512*2**20, 20*2**30))
                    record = dict(row, footprints=[footprint(data / f"s{s:04d}", s) for s in task["sectors"]])
                    # Initial scheduling assumes equal recovery probability; later
                    # observed recovery and runtime can replace this coarse estimate.
                    seconds = 120 + 120*len(task["sectors"]) + 180*len(task["injections"])
                    record["scheduling"] = dict(estimated_seconds=seconds,
                        unassessed_models=len(task["injections"]),
                        expected_added_weight_per_hour=.5*.2/21740*len(task["injections"])/(seconds/3600),
                        method="equal-recovery-runtime-estimate-v1")
                    write_json(target, record, private=True)
            ready.append(task["task_id"])
        except Exception as error:
            failures.append(dict(task_id=task["task_id"], reason=str(error)[:300]))
        write_json(destination / "status.json", dict(ready=ready, failures=failures), private=True)
    return ready, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    ready, failures = prepare(args.inventory, read_json(args.ledger), args.destination)
    print(json.dumps(dict(ready=len(ready), failed=len(failures))))


if __name__ == "__main__":
    main()
