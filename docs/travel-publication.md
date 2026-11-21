# Publishing travel reviews

`scripts/publish_travel_reviews.py OUTPUT_DIRECTORY` runs after independent
review has accepted a submission. Use the review environment, with `p9search`
installed. This is an operator command; participants do not run it.

The output directory must be directly under `outputs/`. Its private
`config.json` specifies the results directory, an isolated `main` checkout of
`oklo/planet-nine-search`, the frozen reference catalog and its SHA-256,
approved scientific source hashes, the installed review source hash, and hashes
of the accounting implementations. `base` identifies the last incorporated
result ID, the previous frozen ledger and summary hash, and the public snapshot
hash. The initial baseline must agree with both GitHub and oklo.org.

The publisher checks the signed review against the coordinator's stored
acceptance, binds reference IDs to the frozen catalog and declared brightness,
then applies the existing field ledger. Full replay is required for candidate
fields. Such fields retain their candidates and unassessed injections without
credit. Image-quality rejections receive neither searched-field nor recovery
credit. Previously accounted or pending reference IDs stop accounting.

Each ledger is written atomically. Publication records the previous and next
snapshot, commits the next version to GitHub, and conditionally replaces the
website snapshot. It advances `head.json` only after HTTPS confirms the expected
hash. A transport failure retries the same result; it does not repeat credit.
An unexpected change writes `hold.json` for operator review. Do not delete a
hold until its evidence has been reconciled.

Run this command with a 60-second launchd `StartInterval`, without `KeepAlive`.
The advisory lock prevents overlapping runs. `STOP` disables work. Health and
errors replace small JSON records; no continuous log is required. Completed
local snapshot copies are removed after checkpointing; GitHub retains those
versions. Frozen ledgers and signed review evidence remain available locally.
Server-side backups are retained.

This process publishes accepted searches. Candidate follow-up decisions require
their separate scientific review and cannot be inferred from acceptance alone.
