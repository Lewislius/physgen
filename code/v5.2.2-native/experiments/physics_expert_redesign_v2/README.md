# Physics Expert redesign v2: R17-A/B/C

Implemented variants:

- `p1_request_key` = R17-A: R10A full-width enhancement + shared full-width
  `RequestKeyEnhancer`, original `C_execute`, full-width direct physical read,
  state-value retained, native text Q/O LoRA16, native self/FFN LoRA32.
- `p2_pre_self` = R17-B: R10A full-width direct read, no state-value, full-width
  physical write before the native block, native self/FFN LoRA64, text Q/O LoRA16.
- `p4_source_conditioned_experts` = R17-C: R10A full-width direct read,
  no state-value, post-self direct write, two **full-width** source-conditioned
  experts (`3072 -> 3072` state/source/output; no `3072 -> 512 -> 3072` path),
  native self/FFN LoRA32, text Q/O LoRA16.

All variants use `joint400`: 400 updates, accumulation 4, all selected parameters
from update 1, fresh official foundations, seed42, 50-step shifted Euler at
shift 5, CFG5. Training and generation use the same forward.

The P4 correction increased trainable parameters from approximately 676M to
1,148M because the low-dimensional expert was replaced by full-width maps. Its
corrected run is the `p4_s3_v5` run; the earlier `p4_s3_v4` run was stopped and
is not a result.
