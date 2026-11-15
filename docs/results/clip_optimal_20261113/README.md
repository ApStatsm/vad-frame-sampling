# CLIP cached-feature sampling results — 2026-11-13

UCF-Crime and XD-Violence full sweeps completed: 37 successful runs each.
Every UCF run evaluated 290 videos; every XD run evaluated 800 videos.
Cases: Reference 100%, Uniform at 100/50/25/12.5%, Random-feature at those
four budgets with seeds 0–4, and D/A/E-optimal at those four budgets.

- `ucf/comparison.csv`, `xd/comparison.csv`: final comparison tables.
- Each dataset's `summary.json`: original comparison metadata and aggregated rows.
- Each dataset's `sweep.json`: original completion records, run paths and original summary hashes.
- `runs/*.json`: run summaries with only the top-level `videos` array removed.
  These compact copies retain metrics, configuration, environment and source hashes;
  their bytes differ from the originals referenced by the sweep hashes.

Original full run summaries and per-video artifacts remain under
`/workspace/hyeongjun/vad/outputs/runs/`. Original comparison artifacts remain
under that project's `outputs/analyses/`; paths in archived JSON refer to that server.

These are cached-feature exploratory performance experiments. All reference CLIP
features were already computed; these results do not demonstrate raw-video speedup.
The runs record a dirty source checkout. This results-only commit does not include
the uncommitted implementation; use recorded source hashes when checking provenance.

Completion was checked from saved JSON and tmux output, without rerunning inference,
tests or aggregation. Detailed numerical interpretation remains pending.

Archived CSV line endings are normalized to LF; cell contents are unchanged.

The subsequent implementation commit archives the experiment code, settings,
comparison tools and tests. Before committing, all 25 `vadfs` source hashes
recorded across the 74 run summaries were checked against the working files;
all matched. This establishes package-source identity, without rerunning the
experiments. The historical `research_commit` and dirty flags remain unchanged.
