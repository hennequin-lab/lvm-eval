# NLB dataset protocol reference

This file records the dataset-level values used by NLB'21. It is **documentation
for callers**, not a set of code defaults or presets. `lvm-eval` has no dataset
configs: callers pass tensors and parameters explicitly. Metric-level defaults
(the 1e-9 rate floor, the 9-alpha ridge grid, 5-fold CV) are part of the metric
and live in code; dataset-level values live here.

Sources:

- paper: `/home/fh397/projects/distil/_papers/NLB.pdf` §3.1, §A.5.1, §A.5.2,
  §A.5.3, §A.6;
- `nlb_tools`: `/home/fh397/projects/_refs/nlb_tools`, commit
  `42f8410b88e12db136910fa2f888b025ea0aa2ae`, especially
  `nlb_tools/make_tensors.py` (`PARAMS`, lines 14-204) and
  `nlb_tools/evaluation.py`.

Tensor layout for all values below is `[trial; time; neuron]`, matching
`nlb_tools`. Rates are expected spike counts per bin (Poisson means), not Hz.
The reference EvalAI submission bin width is 5 ms; `nlb_tools` also evaluates
20 ms submissions, but bin width is not a parameter of the metric itself.

## Core values

| Dataset | Bin width | Trial alignment / eval window | Decode lag | PSTH kernel σ | Forward window |
|---|---:|---|---:|---:|---|
| MC_Maze | 5 ms | movement onset, −250…450 ms | 100 ms | 70 ms | 450…650 ms after movement onset (200 ms) |
| MC_RTT | 5 ms | snippet start, 0…600 ms; overlap allowed | 140 ms | none (no PSTH) | 600…800 ms after snippet start (200 ms) |
| Area2_Bump | 5 ms | movement onset, −100…500 ms; eval overlap allowed | −20 ms | 40 ms | 500…700 ms after movement onset (200 ms) |
| DMFC_RSG | 5 ms | eval: set_time…go_time, `seg_len=1500` ms, pad front; align to go_time | none in `PARAMS`; speed–`tp` metric does not use a lag | 70 ms | 0…200 ms after go_time (200 ms) |
| MC_Maze-L | 5 ms | movement onset, −250…450 ms | 120 ms | 50 ms | 450…650 ms after movement onset (200 ms) |
| MC_Maze-M | 5 ms | movement onset, −250…450 ms | 120 ms | 50 ms | 450…650 ms after movement onset (200 ms) |
| MC_Maze-S | 5 ms | movement onset, −250…450 ms | 120 ms | 50 ms | 450…650 ms after movement onset (200 ms) |

Citations:

- Trial alignment: paper §A.5.1; `make_tensors.py:21-28`, `:47-56`,
  `:68-76`, `:98-111`, `:134-141`, `:160-167`, `:186-193`.
- Decode lag: paper §A.5.2; `PARAMS` `lag` fields at
  `make_tensors.py:20`, `:46`, `:67`, `:133`, `:159`, `:185`. DMFC_RSG has no
  `lag` key in `PARAMS`; paper §A.5.2 also omits DMFC_RSG.
- PSTH kernel σ: paper §A.5.3; `PARAMS` `psth_params.kern_sd` at
  `make_tensors.py:36`, `:84`, `:120`, `:149`, `:175`, `:201`. The paper notes
  MC_Maze-L’s optimal value was 40 ms but 50 ms is used to match MC_Maze-M/S.
- Forward window: `_prep_fp` at `make_tensors.py:1087-1115`; `fp_len=200` at
  `make_tensors.py:29`, `:57`, `:77`, `:112`, `:142`, `:168`, `:194`. The
  window starts at the end of the main `align_range` when present, otherwise
  at 0 relative to the alignment point; for DMFC_RSG the eval params have no
  `align_range`, so the forward window is 0…200 ms after go_time. The paper
  explicitly mentions the extra 200 ms for MC_RTT; the reference applies
  `fp_len=200` to every dataset.

## Dataset-specific fields

| Dataset | PSTH condition fields | Jitter | Decode masks | Behaviour field |
|---|---|---|---|---|
| MC_Maze | `trial_type`, `trial_version` | zero | none (one all-True mask in `evaluate`) | `hand_vel` |
| MC_RTT | none (no PSTH) | zero | none | `finger_vel` |
| Area2_Bump | `cond_dir`, `ctr_hold_bump` | zero | `ctr_hold_bump == 0`, `ctr_hold_bump == 1` (two decoders) | `hand_vel` |
| DMFC_RSG | `is_eye`, `theta`, `is_short`, `ts` | test: 0. Non-test: uniform in `[0, clip(1500 − tp, 0, 300)]` ms, rounded to integer bins. | none | trial_info fields `is_eye`, `theta`, `is_short`, `ts`, `tp` |
| MC_Maze-L/M/S | `trial_type`, `trial_version` | zero | none | `hand_vel` |

Citations:

- PSTH condition fields: `PARAMS` `psth_params.cond_fields` at
  `make_tensors.py:31`, `:79`, `:114`, `:144`, `:170`, `:196`.
- DMFC jitter: `PARAMS` `jitter` at `make_tensors.py:93-97`;
  `_prep_jitter` at `make_tensors.py:1118-1156`; stored in bins at
  `make_tensors.py:665`. The jitter is applied to eval rate trials by
  `evaluation.py:355-361` before condition averaging.
- Area2_Bump decode masks: `PARAMS` `decode_masks` at
  `make_tensors.py:66`; masks are `[trial, n_masks]` bool and
  `velocity_decoding` fits one ridge per column (`evaluation.py:479-491`).
- Behaviour fields: `PARAMS` `behavior_field` at `make_tensors.py:19`,
  `:45`, `:65`, `:92`, `:132`, `:158`, `:184`. DMFC behaviour comes from
  `trial_info` (`behavior_source='trial_info'`, `make_tensors.py:90-92`) and
  `tp` is the last column; `speed_tp_correlation` uses columns `[0,1,2]` for
  conditions and the last column for `tp` (`evaluation.py:422-429`).
- DMFC eval tensor: `eval_tensor_params` `seg_len=1500`, `pad='front'` at
  `make_tensors.py:108-111`; PSTH uses the same padding/length at
  `make_tensors.py:121-122`.

## Reference notes for callers

- **PSTH is caller-built.** `lvm-eval` accepts a precomputed `psth` plus
  `eval_cond_idx` and `jitter`. The reference builds it at 1 ms
  (`nwb_interface.py:623 smooth_spk`) and then resamples to the metric bin
  (`nwb_interface.py:548 resample`; `make_tensors.py:1202-1211`). Smoothing
  already-binned 5 ms spikes is not equivalent. The caller also owns trial
  exclusions, the 0.8× mean-length short-trial drop, truncation/padding, and
  condition definitions (`make_tensors.py:1241-1264`).
- **Jitter is caller-supplied in integer bins.** The values above describe the
  reference's jitter distribution; `lvm-eval` does not generate jitter and does
  not reproduce NumPy’s RNG.
- **No dataset configs in code.** These values are not compiled into
  `lvm-eval` and are not used to select data. A caller that wants
  NLB-comparable numbers must supply the matching tensors, masks, conditions,
  jitter, lag-aligned behaviour, and PSTH.
- **The DMFC global variant is not in `nlb_tools`.** The dataset values above
  describe the `nlb_tools` per-condition variant. The global variant uses the
  same per-trial speed and set-go mask but one Pearson r across all eval
  trials; its goldens come from a separate Python script. See `plan.md` §4.5.
- **5 ms vs 20 ms.** The paper describes 5 ms submissions; `nlb_tools` also
  evaluates 20 ms submissions (`evaluation.py:14-207` loops over `[5, 20]`).
  Bin width is data preparation; `lvm-eval` metrics consume per-bin rates and
  do not take a bin-width argument.

## Related documents

- [`../README.md`](../README.md) — short library overview.
- [`../plan.md`](../plan.md) — API, metric specifications, verification,
  milestones, edge cases, and known deviations.
