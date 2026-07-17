# `data/tables/`

- `dataset_graphene.csv` — ground-truth physical properties (`total_energy`,
  lattice parameters) computed by the original atomistic simulations. This is
  the label source `dataset.merge_energy_labels` reads.
- `box_dataframe.csv`, `groupedDataframe.csv`, `energy_dataframe.csv`,
  `no_name_dataframe.csv` — **historical outputs** from the original thesis
  notebook run, kept as a reference baseline (they are what
  `docs/known_quirks.md`'s numeric-parity check was validated against).
  Re-running `gdd build-dataset` regenerates equivalent, traceable files
  (`box_dataframe.csv`, `grouped_dataframe.csv`, `energy_dataframe.csv`) in
  this same directory with an explicit `sample_id` column — see
  `docs/known_quirks.md` §3 for why that matters.
