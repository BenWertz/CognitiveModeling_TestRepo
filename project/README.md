## What Is Here

- `data/All_Data_Visual.csv`: visual recall data.
- `models/`: the four selected Stan models plus compiled executables.
- `scripts/`: refit scripts for the four selected models plus the report-bundle
  builder.

## Selected Models

| Model name | Fit script |
| --- | --- |
| `condition_sliding_frame` | `scripts/fit_condition_sliding_frame.py` |
| `hierarchical_sliding_frame` | `scripts/fit_hierarchical_sliding_frame.py` |
| `hierarchical_sliding_frame_target_precision` | `scripts/fit_hierarchical_sliding_frame_target_precision.py` |
| `hierarchical_no_swap_target_precision` | `scripts/fit_hierarchical_no_swap_target_precision.py` |


## Refit Models

Run commands from this folder:

```powershell
python .\scripts\fit_condition_sliding_frame.py
python .\scripts\fit_hierarchical_sliding_frame.py
python .\scripts\fit_hierarchical_sliding_frame_target_precision.py
python .\scripts\fit_hierarchical_no_swap_target_precision.py
```
