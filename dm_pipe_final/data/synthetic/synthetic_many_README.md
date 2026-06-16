# Synthetic mismatch sessions - expanded scenario set

Generated from `now/out/minute_model.csv` using normal-ish source sessions selected from real sessions with enough length, low zero-chat rate, positive chat mean, and low review priority.

## Files
- `synthetic_many_minutes.csv`: injected synthetic minute rows, 77,360 rows, 640 synthetic sessions.
- `synthetic_many_intervals.csv`: injection interval metadata, 640 rows.
- `synthetic_many_summary.csv`: before/after summary by synthetic session.
- `synthetic_many_source_controls.csv`: source-control duplicate rows, 4,835 rows, 40 control sessions.

## Scenario groups
| injection_type                | injection_family               | expected_recovery   |   n |
|:------------------------------|:-------------------------------|:--------------------|----:|
| chat_collapse_only            | positive_chat_deficit          | positive            |  40 |
| early_silence                 | positive_position_early        | positive            |  40 |
| gradual_view_inflate_low_chat | positive_gradual_deficit       | positive            |  40 |
| hi_view_low_chat_mild         | positive_deficit               | positive            |  40 |
| hi_view_low_chat_strong       | positive_deficit               | positive            |  40 |
| intermittent_zero_control     | negative_control_noncontiguous | negative_control    |  40 |
| late_silence                  | positive_position_late         | positive            |  40 |
| scattered_low_chat_control    | negative_control_noncontiguous | negative_control    |  40 |
| silent_run_long               | positive_contiguous_silence    | positive            |  40 |
| silent_run_medium             | positive_contiguous_silence    | positive            |  40 |
| silent_run_short              | positive_contiguous_silence    | positive            |  40 |
| slow_chat_decay               | positive_gradual_deficit       | positive            |  40 |
| three_block_dropout           | difficult_multiblock           | difficult           |  40 |
| unique_collapse_repeated_chat | positive_unique_deficit        | positive            |  40 |
| view_spike_no_chat_short      | difficult_view_spike           | difficult           |  40 |
| viewer_spike_then_silence     | positive_spike_then_silence    | positive            |  40 |

## Important interpretation
These files are **not real viewbot labels** and **not performance results**. They are synthetic viewer-chat mismatch injection inputs for sanity checking whether the pipeline recovers continuous mismatch intervals.
Use them under `now/data/synthetic/` and run the robustness/synthetic evaluation. Do **not** place them under `data/features/` or `data/labels/`.

Recommended placement:
```
now/data/synthetic/synthetic_many_minutes.csv
now/data/synthetic/synthetic_many_intervals.csv
now/data/synthetic/synthetic_many_summary.csv
now/data/synthetic/synthetic_many_source_controls.csv
```

Negative/difficult scenarios:
- `intermittent_zero_control`, `scattered_low_chat_control` are negative-control-like and should not be counted as positive synthetic recovery targets.
- `view_spike_no_chat_short` and `three_block_dropout` are difficult cases. Failure to recover them means the current pipeline is focused on persistent chat/unique deficit rather than short viewer spikes or fragmented mismatch blocks.
