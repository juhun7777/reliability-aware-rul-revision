# Changes from the original implementation

This separate revision repository derives from `juhun7777/reliability-aware-rul`
original main commit `d2276a48ce004627e7d9c29df7884fc89565c693`, incorporating the
validated revision code from commit `143f99a0a85802b2ba835924928450fbd54d3146`.
The original repository is not changed by this publication.

## Correctness and traceability

- Remove condition-specific test-file replacement; use stable splits and declared observations.
- Use measured cumulative cycles, not RUL-derived differences.
- Split STFT feature groups before fitting scalers; clone and restore best weights.
- Use an observed initial PHI baseline; retain an explicit historical calculator mode.
- Export file/group/observation identity, expert outputs, weights, checkpoints,
  scaler, configuration, seed and input hashes.
- Make perfect-prediction NASA score zero and document its scaled/log definition.

## Separate revision workflow

- Explicit manifest, original-parent group checks, training-only epoch/gate selection.
- Pooled, independent-expert and shared-encoder Transformer controls.
- Matched soft/hard/uniform combinations, fixed pooled blend and PHI control.
- Repeatable MC dropout with exported draws and independent inference seed.
- Checkpoint replay, matched spectral-bin control, internal mixing calibration,
  and calibration/noise/table evidence tools.

## Public contents and validation

The 18 sample CSV files and two illustrations are byte-identical to the
already-public source. Full private measurement sets, checkpoints, local reports,
credentials and development result folders are not included.

The core revision code passed 13 unit tests, an 18-CSV smoke run and saved
checkpoint replay during development. The two-epoch sample command checks code
execution only; it does not reproduce manuscript performance. No final journal
publication status or historical numeric claims are asserted by this repository.
