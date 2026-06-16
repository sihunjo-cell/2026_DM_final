# Synthetic injection example sessions

These are **example synthetic viewer-chat mismatch injection rows** created from low-review-order/high-chat real sessions in the provided pipeline output. They are not real viewbot labels and should not be mixed into training. Use them only to test the proposed evaluation code path.

Files:
- `synthetic_injection_example_minutes.csv`: minute-level rows in the same style as `minute_model.csv`, with synthetic metadata columns.
- `synthetic_injection_example_intervals.csv`: injected interval metadata.
- `synthetic_injection_example_summary.csv`: before/after summary for each example session.

Scenarios:
- `silent_run`: viewer kept, chat/unique set to zero in a contiguous interval.
- `hi_view_low_chat`: viewer inflated and chat/unique suppressed in a contiguous interval.
- `view_spike_no_chat`: short viewer spike with weak chat/unique response.
- `intermittent_zero_control`: scattered zero-chat minutes, negative-control-like non-contiguous case.

Important: fit baseline/KMeans/aggregation on real data only, then transform/score these synthetic rows. Do not report metrics from these examples as real viewbot performance.
