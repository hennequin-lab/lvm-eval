# lvm-eval implementation plan

Status: approved design, no implementation yet. This plan is the source of
truth for the first release. `README.md` is the short user-facing summary and
`docs/nlb-protocol.md` is the dataset-value reference. If this plan and an
older README proposal disagree, this plan wins.

Real paths in this repository's environment:

- paper: `/home/fh397/projects/distil/_papers/NLB.pdf`
- reference implementation: `/home/fh397/projects/_refs/nlb_tools`
- murder: `/home/fh397/projects/murder`
- distil: `/home/fh397/projects/distil`

Tensor layout for every public metric is **`[trial; time; neuron]`**. This is
the `nlb_tools` layout. It is not the `[time; trial; neuron]` layout used by
the distil model internally. Rates are expected spike counts per bin (Poisson
means), not Hz. The library computes in float64 on the host CPU and the public
API takes `Nx.float64_t` tensors (see §11 and §8).

## 1. Scope and non-goals

### Scope

- A pure OCaml library of evaluation metrics for latent variable models of
  neural population activity.
- The first release implements the NLB'21 metrics:
  - co-smoothing bits/spike (primary);
  - forward-prediction bits/spike;
  - PSTH R²;
  - DMFC_RSG speed–`tp` Pearson r, both the `nlb_tools` per-condition variant
    and a documented global variant;
  - velocity-decoding R².
- Metrics receive model outputs and ground-truth targets as tensors. They
  return scalar headline values only in the first release.
- For valid inputs, the metrics reproduce `nlb_tools` at commit
  `42f8410b88e12db136910fa2f888b025ea0aa2ae`. Deliberate deviations are
  limited to degenerate inputs and are listed in §9.
- Numerics use OCaml Raven only, at the versions pinned by murder. The
  library computes in float64 on the host CPU.

### Non-goals

- No models, weights, checkpoints, training code, optimisers, or inference.
- No data loading, trial splitting, trial alignment, binning, resampling,
  trial selection, or dataset configs.
- No dataset presets in code. Dataset values live in
  `docs/nlb-protocol.md` as documentation for callers.
- No file I/O, CLI, serialisation, or manifests in the first release. I/O and
  a CLI are later roadmap items only; they would live outside the core metric
  modules.
- No per-neuron or per-trial breakdowns in the first release.
- No latent metrics, probe metrics, leaderboard, or official NLB evaluator.
- No dependency on distil or any other caller.
- No Python in the build or runtime. Python is only used outside dune to
  generate test fixtures under `reference/`.

## 2. Architecture

### 2.1 Dependency rule

`lvm-eval` depends only on Raven (`nx`) and the OCaml standard library/Base
as needed. It never depends on `distil`, or any other project.
Callers depend on `lvm-eval`; the direction is one-way.

`lvm-eval` contains no data loading, splitting, alignment, binning, storage,
or dataset configuration. It has no file I/O, no global state, and no model
types. Callers own all data preparation.

### 2.2 Planned library layout

```text
lib/
  lvm_eval.ml/.mli       public entry point; re-exports metric modules
  check.ml/.mli          rank/shape and precondition checks; clear errors
  mask.ml/.mli           valid-entry counts and masked sums/means
  numeric.ml/.mli        float64 helpers; zero-rate floor on a copy
  r2.ml/.mli             per-output R² with the reference's force-finite rule
  pearson.ml/.mli        sample Pearson r, with explicit undefined handling
  ridge.ml/.mli          centered ridge solve, alpha grid, contiguous K-fold
  bits.ml/.mli           co-smoothing and forward-prediction bps
  psth.ml/.mli           PSTH R² from a precomputed PSTH
  dmfc.ml/.mli           DMFC_RSG speed–tp, code and global variants
  decode.ml/.mli         velocity decoding R²
test/
  test_bits.ml
  test_psth.ml
  test_dmfc.ml
  test_decode.ml
  fixtures/              generated plain arrays and expected values
reference/
  generate_fixtures.py   outside dune; Python reference generator only
```

`check`, `mask`, `numeric`, `r2`, `pearson`, and `ridge` are shared helpers,
not hypothetical future abstractions. No other abstraction is added unless
this plan is amended.

### 2.3 Public/private modules

Only `Lvm_eval.Bits`, `Lvm_eval.Psth`, `Lvm_eval.Dmfc`, and
`Lvm_eval.Decode` are public metric entry points. The shared helpers are
internal implementation modules (`lib/*.ml` without public re-export) unless a
later plan explicitly exposes them. `Lvm_eval` may re-export the metric
modules only.

### 2.4 Shape and dtype contract

- Every tensor argument is float64 or bool as named in §3. Callers with
  float32 cast once at the boundary; the metric core never computes in
  float32.
- Rank and shape compatibility are checked. The library does not validate
  that trial indices, time axes, or neurons are aligned between arguments;
  the caller owns alignment.
- A mismatch raises `Invalid_argument` naming the function, the argument, the
  expected shape, and the shape received.
- No function mutates an input tensor.

## 3. Public API sketch

The signatures below are the intended public interface. They are illustrative
of the plan, not existing code.

```ocaml
module Bits : sig
  val co_smoothing :
    rates_heldout:Nx.float64_t ->
    spikes_heldout:Nx.float64_t ->
    ?mask_heldout:Nx.bool_t ->
    unit -> float

  val forward_prediction :
    rates_forward:Nx.float64_t ->
    spikes_forward:Nx.float64_t ->
    ?mask_forward:Nx.bool_t ->
    unit -> float
end

module Psth : sig
  val r2 :
    rates_eval:Nx.float64_t ->
    psth:Nx.float64_t ->
    eval_cond_idx:int array array ->
    ?jitter:int array ->
    unit -> float
end

module Dmfc : sig
  val speed_tp_code :
    rates_eval:Nx.float64_t ->
    mask:Nx.bool_t ->
    conditions:int array ->
    tp:float array ->
    unit -> float

  val speed_tp_global :
    rates_eval:Nx.float64_t ->
    mask:Nx.bool_t ->
    tp:float array ->
    unit -> float
end

module Decode : sig
  val velocity_decoding :
    rates_train:Nx.float64_t ->
    behaviour_train:Nx.float64_t ->
    rates_eval:Nx.float64_t ->
    behaviour_eval:Nx.float64_t ->
    ?train_decode_mask:Nx.bool_t ->
    ?eval_decode_mask:Nx.bool_t ->
    ?alphas:float array ->
    ?folds:int ->
    unit -> float
end
```

### 3.1 Argument meanings and shapes

All shapes are `[trial; time; neuron]` unless stated otherwise.

| Function | Argument | Role | Shape / meaning |
|---|---|---|---|
| `Bits.co_smoothing` | `rates_heldout` | held-in-derived model output | `[B,T,N_ho]`; predicted Poisson means for held-out neurons |
| | `spikes_heldout` | held-out target | `[B,T,N_ho]`; observed counts; NaN may mark excluded entries when no mask is supplied |
| | `mask_heldout` | optional metadata | `[B,T,N_ho]` bool; if absent, valid = non-NaN spikes |
| `Bits.forward_prediction` | `rates_forward` | held-out model output for future bins, all neurons | `[B,F,N_all]`; predicted Poisson means |
| | `spikes_forward` | held-out target | `[B,F,N_all]`; future counts |
| | `mask_forward` | optional metadata | `[B,F,N_all]` bool |
| `Psth.r2` | `rates_eval` | held-out model output | `[B,T,N_all]`; eval-trial rates |
| | `psth` | held-out target | `[C,T_psth,N_all]`; precomputed PSTH, NaN-padded |
| | `eval_cond_idx` | metadata | `int array array` length `C`; trial indices per condition |
| | `jitter` | metadata | `int array` length `B`; shifts in bins; default all zeros |
| `Dmfc.speed_tp_code` | `rates_eval` | held-out model output | `[B,T,N_all]`; eval-trial rates, all neurons |
| | `mask` | metadata | `[B,T]` bool; set-go time points |
| | `conditions` | metadata | `int array` length `B`; condition label per trial |
| | `tp` | held-out target | `float array` length `B`; produced interval in seconds |
| `Dmfc.speed_tp_global` | `rates_eval` | held-out model output | `[B,T,N_all]` |
| | `mask` | metadata | `[B,T]` bool |
| | `tp` | held-out target | `float array` length `B` |
| `Decode.velocity_decoding` | `rates_train` | model output on training trials | `[B_tr,T,N_all]` |
| | `behaviour_train` | training target | `[B_tr,T,D]`; D=2 velocity |
| | `rates_eval` | held-out model output | `[B_ev,T,N_all]` |
| | `behaviour_eval` | held-out target | `[B_ev,T,D]` |
| | `train_decode_mask` | metadata | `[B_tr,K]` bool; default one all-True column |
| | `eval_decode_mask` | metadata | `[B_ev,K]` bool; default one all-True column |
| | `alphas` | metric parameter | default `10. ** (-4. +. 0.5 *. k)` for `k=0..8` |
| | `folds` | metric parameter | default 5; contiguous folds over flattened samples |

`B` = trials, `T` = time bins, `N_ho` = held-out neurons, `N_all` = all
neurons (held-in + held-out), `F` = future bins, `C` = conditions,
`D` = behaviour dimensions, `K` = decode masks.

### 3.2 Examples: how distil would call each metric

Distil’s current arrays are `[time; trial; neuron]` and its rates are Hz.
The caller transposes and converts to counts/bin before calling.

```ocaml
(* After distil has a held-out-neuron model. dt = 0.005 s. *)
let rates_t_t_n =
  let x = Nx.cast Nx.float64 rates_hz |> Nx.transpose ~axes:[1;0;2] in
  Nx.Infix.(x *$ 0.005)
in
let bps =
  Lvm_eval.Bits.co_smoothing
    ~rates_heldout:(heldout_neurons rates_t_t_n)
    ~spikes_heldout:(heldout_neurons (Nx.cast Nx.float64 spikes_t_t_n))
    ~mask_heldout:(heldout_neurons valid_mask_t_t_n)
    ()
```

The other examples below assume the caller has already cast float tensors to
float64 and transposed them to `[trial; time; neuron]`.

```ocaml
(* Forward is blocked in distil until the extra future window exists. *)
let fp_bps =
  Lvm_eval.Bits.forward_prediction
    ~rates_forward:forward_rates_t_f_n
    ~spikes_forward:forward_spikes_t_f_n
    ()
```

```ocaml
(* PSTH is precomputed by the caller, at 1 ms then resampled. *)
let psth_r2 =
  Lvm_eval.Psth.r2
    ~rates_eval:rates_t_t_n
    ~psth:psth_c_t_n
    ~eval_cond_idx:condition_trial_indices
    ~jitter:jitter_bins
    ()
```

```ocaml
let code_r =
  Lvm_eval.Dmfc.speed_tp_code
    ~rates_eval:rates_t_t_n
    ~mask:set_go_t_t
    ~conditions:condition_labels
    ~tp:produced_intervals_s
    ()

let global_r =
  Lvm_eval.Dmfc.speed_tp_global
    ~rates_eval:rates_t_t_n
    ~mask:set_go_t_t
    ~tp:produced_intervals_s
    ()
```

```ocaml
let decode_r2 =
  Lvm_eval.Decode.velocity_decoding
    ~rates_train:rates_train_t_t_n
    ~behaviour_train:velocity_train_t_t_d
    ~rates_eval:rates_eval_t_t_n
    ~behaviour_eval:velocity_eval_t_t_d
    ~train_decode_mask:train_masks
    ~eval_decode_mask:eval_masks
    ()
```

## 3b. METRIC/CALLER boundary

### Co-smoothing

- **Inside lvm-eval (METRIC):** shape checks; valid-entry calculation from an
  explicit mask or non-NaN spikes; finite/nonnegative rate checks; zero-rate
  floor `1e-9` on a copy; per-neuron null mean over valid entries; Poisson NLL
  for model and null; `(NLL_null - NLL_model)/(sum spikes · log 2)`.
- **Caller (CALLER):** loading; held-in/held-out neuron split; trial/time
  alignment; 5 ms binning; model inference; converting Hz to counts/bin;
  padding; explicit masks; choosing held-out neurons and test trials.

### Forward prediction

- **Inside lvm-eval (METRIC):** the same bps computation, with the null mean
  over the valid future entries.
- **Caller (CALLER):** choosing the future window (reference: 200 ms starting
  after the main evaluation window); extracting future spikes; building
  all-neuron forward rates/spikes; alignment and padding; per-bin rates.

### PSTH R²

- **Inside lvm-eval (METRIC):** jitter application; per-condition averaging of
  eval rates; first-neuron-NaN row selection; finite-prediction check; pooling
  condition-time rows; per-neuron R²; average across neurons.
- **Caller (CALLER):** PSTH construction from spikes at 1 ms then resampling
  to the metric bin; condition definitions; trial exclusions; short-trial
  handling; padding; kernel σ; jitter values. `lvm-eval` receives the finished
  `psth`, `eval_cond_idx`, and `jitter`.

### DMFC_RSG speed–tp

- **Inside lvm-eval (METRIC):** apply mask; per-trial speed = mean over time
  of the L2 norm of the time difference across neurons; per-condition Pearson
  r for `speed_tp_code`; one global Pearson r for `speed_tp_global`;
  aggregation.
- **Caller (CALLER):** set-go mask; condition labels; `tp`; all-neuron eval
  rates; alignment; deciding code vs global variant. The global variant is a
  documented minimal reading of the paper, not an `nlb_tools` reproduction.

### Velocity decoding

- **Inside lvm-eval (METRIC):** select the trials chosen by the mask; flatten
  trial-major; drop flattened samples with NaN in the first behaviour column;
  require finite retained rates/behaviour; fit one centered ridge per mask
  column; select alpha by contiguous K-fold CV; predict; per-output R²;
  average across masks.
- **Caller (CALLER):** train/eval split; behaviour lag and alignment; decode
  masks (Area2_Bump active/passive); behaviour extraction and units; padding;
  per-bin rates; official split; any feature standardization (the NLB path
  does none).

## 4. Per-metric specification

### 4.1 Co-smoothing bits/spike

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.2:

```text
bps = 1/(n_sp log 2) [ L(λ; ŷ) - L(λ̄; ŷ) ]
L(λ; y) = Σ [ y log λ - λ ]        (up to the constant log y!)
```

`λ` is the predicted Poisson mean per bin, `λ̄` is the per-neuron mean spike
count per bin over the scored entries, and `n_sp` is the total number of
scored spikes. The equation is unnumbered in the paper; it is the displayed
equation in §3.2.

**Reference.** `evaluation.py:209 neg_log_likelihood`, `:252 bits_per_spike`.
The reference uses `NLL = Σ[λ - y log λ + lgamma(y+1)]`, a null rate from
`np.nanmean` over all axes except the neuron axis, and denominator
`np.nansum(spikes) * log 2`. Zero rates are replaced by `1e-9` in place.

**Numerical approach.** Inputs are float64 at the API boundary; build a valid mask (explicit mask or
`not isnan spikes`); per neuron compute `sum(valid * spikes)` and
`count(valid)`; null rate = sum/count; floor exact zero null rates to `1e-9`
on a copy; compute `Nx.lgamma(spikes + 1)` on valid entries; sum in float64;
`bps = (NLL_null - NLL_model) / (sum_spikes * Float.log 2.)`.

**Pitfalls.**
- Use the global spike total for `n_sp`; the paper’s “its total number of
  spikes” is per-neuron wording but the code normalizes globally.
- The null mean is per neuron, not per time or per trial, and uses that
  neuron’s own valid count.
- Do not mutate the rate input when flooring zeros.
- NaN rates on valid entries are an error; NaNs outside the mask are ignored.
- Shape must be rank 3; last axis is neurons.
- `Nx.sum` order is unspecified; compare with relative tolerance.

### 4.2 Forward-prediction bits/spike

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.3: same bps as
co-smoothing, with held-out entries at future time points across all neurons.

**Reference.** `evaluation.py:14 evaluate` (forward branch) and
`:252 bits_per_spike`. The reference reads held-in and held-out forward spikes
and rates, concatenates them along the neuron axis with `np.dstack`, and calls
`bits_per_spike`. The future window is built by `make_tensors.py:1087
_prep_fp`; `fp_len=200` ms for every dataset in `PARAMS`.

**Numerical approach.** Reuse the `Bits` bps core unchanged. The only
difference is the input tensors and their shape `[B,F,N_all]`.

**Pitfalls.**
- All neurons, not just held-out neurons.
- The null mean is over the forward window only.
- The window start and length are caller data prep; the metric has no window
  parameter.
- Do not claim a distil forward result: current distil data has no extra
  future window.

### 4.3 PSTH R²

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.3 and A.5.3, A.6:
trial-averaged model rates per condition compared to an empirical PSTH built
from smoothed spikes; R² per neuron across conditions, then averaged across
neurons. The reference evaluator consumes a precomputed `psth`; PSTH
construction is in `make_tensors.py:1158 _make_psth` and
`nwb_interface.py:623 smooth_spk`, `:548 resample`.

**Reference.** `evaluation.py:332 eval_psth`. It jitters each eval rate trial
by an integer number of bins, averages within condition, selects rows where
the first PSTH neuron is not NaN, stacks condition-time rows, and calls
`r2_score` (uniform average over neurons).

**Numerical approach.** Inputs are float64 at the API boundary. For each condition with at least one
eval trial: apply `jitter_trial` exactly (positive prepends NaN and drops the
last bins; negative appends NaN and drops the first bins); average the
jittered rates; select rows where `psth[c,:,0]` is not NaN; require the
selected predictions finite; append to row lists. Stack rows; compute R² per
neuron over the pooled condition-time rows; average neurons.

**Pitfalls.**
- `lvm-eval` does **not** construct the PSTH. The caller must smooth at 1 ms
  and resample; smoothing already-binned 5 ms spikes is not equivalent.
- Condition-time rows are pooled; conditions with more valid bins weigh more.
- The first-neuron-NaN padding assumption is reproduced; rows where another
  neuron is NaN are not silently masked.
- `jitter` is in integer bins and is caller-supplied; its length must equal
  the eval trial count.
- `eval_cond_idx` length must equal the PSTH condition count; empty conditions
  are skipped.
- The PSTH time length and the eval-rate time length must match after jitter;
  a mismatch is a clear shape error.
- PSTH values are mean smoothed counts per bin, not Hz.
- The PSTH time length and the eval-rate time length must match; otherwise
  raise a clear shape error.

### 4.4 DMFC_RSG speed–tp, code variant

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.3 and A.6: average
neural speed per trial from test rate predictions, then Pearson r with `tp`.
A.6 specifies the set-go window and jitter.

**Reference.** `evaluation.py:386 speed_tp_correlation`. It derives the set-go
mask from `eval_spikes_heldout[...,0]` NaNs, computes
`mean(norm(diff(rates, axis=0), axis=1))` per trial, groups by condition
columns `[is_eye, theta, is_short]`, computes Pearson r per condition, and
averages the r values.

**Numerical approach.** Require `rates_eval[B,T,N_all]`, `mask[B,T]`,
`conditions[B]`, `tp[B]`. For each trial, select `mask` time points; differ
along time; L2 norm across neurons; mean over time. For each unique condition
label, require at least two trials and nonzero variance in both speed and
`tp`; compute Pearson r; average r across conditions.

**Pitfalls.**
- All neurons, not only held-out neurons.
- The caller supplies the mask; reference derives it from held-out spike NaNs.
- The caller supplies one integer label per trial. For the reference variant,
  encode the tuple `(is_eye, theta, is_short)` from the DMFC behaviour matrix
  as that label; the reference groups by those three columns.
- `tp` is in seconds in the reference (`timedelta.total_seconds()`).
- Singleton or constant conditions are undefined; see §9.

### 4.5 DMFC_RSG speed–tp, global variant

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.3 says “calculate
Pearson’s r between neural speeds and the measured `tp`.” It does not specify
the mask, neuron set, or speed definition.

**Reference.** There is no `nlb_tools` implementation. This variant is a
minimal reading of the paper: same per-trial speed and same set-go mask as
`speed_tp_code`, but one Pearson r across all eval trials, ignoring condition
labels.

**Numerical approach.** Same speed computation as §4.4; one Pearson r over
all trials with finite speed and `tp`. Golden values come from a separate
Python script using the same speed formula and `scipy.stats.pearsonr`; the
script is not a reference implementation of NLB.

**Pitfalls.**
- Do not describe this as a reproduction of `nlb_tools`; it is a documented
  paper-reading variant.
- Condition labels are not used.
- Singleton/constant inputs are undefined; see §9.

### 4.6 Velocity-decoding R²

**Paper.** `/home/fh397/projects/distil/_papers/NLB.pdf` §3.3: ridge
regression from training rates to hand velocity, R² on test rates. A.5.2 gives
decoding lags; the lag is applied by data preparation in the reference.

**Reference.** `evaluation.py:279 fit_and_eval_decoder` and `:437
velocity_decoding`. `velocity_decoding` fits one ridge per decode-mask column,
flattens selected trials/time to samples, and averages R² across columns.
`fit_and_eval_decoder` drops rows by the first behaviour column NaN, asserts
finite rates, fits `GridSearchCV(Ridge(), {"alpha": np.logspace(-4,0,9)})`,
and returns `decoder.score` (R²).

**Numerical approach.** Inputs are float64 at the API boundary. For each mask
column: select the mask trials; flatten `[trial;time;feature]` trial-major;
drop flattened samples with NaN in the first behaviour column; require all
retained rates/behaviour finite. Center `X` and `y`; for each alpha in the
default grid solve `(XᵀX + αI)w = Xᵀy` with `Nx.cholesky` +
`solve_triangular`, falling back to `Nx.svd` if the factorization is not
positive-definite. Contiguous 5-fold over the flattened training samples;
choose the alpha with the highest mean validation R²; refit on all training
samples; predict eval; compute R² per output dimension with sklearn’s
`force_finite` rule; average dimensions and then decode-mask columns.

**Pitfalls.**
- One ridge per mask column, averaged; this contradicts the old README’s “one
  linear map” wording, which is removed.
- No feature standardization and no decoding lag inside the metric.
- The CV folds are over flattened samples, not trials.
- Empty masks, training samples fewer than `folds` (default 5), or non-finite
  retained values are errors. `folds` must be at least 2.
- Constant behaviour dimensions use the force-finite rule: R² 1 if perfect,
  otherwise 0.
- `alphas`/`folds` are metric parameters with NLB defaults; callers may
  override.
- `train_decode_mask` and `eval_decode_mask` must both be present or both be
  absent; if both are absent, use one all-True column. If present, their `K`
  must match.

## 5. Verification strategy

### 5.1 Golden values

Generate goldens outside dune with `reference/generate_fixtures.py` (Python is
fixture-only, never build/runtime). The generator imports
`/home/fh397/projects/_refs/nlb_tools/nlb_tools/evaluation.py` at commit
`42f8410b88e12db136910fa2f888b025ea0aa2ae` and writes plain fixture arrays plus
expected values.

- Co-smoothing: `evaluation.py:252 bits_per_spike`.
- Forward: the same function on forward-shaped tensors; `evaluate`’s `dstack`
  concatenation is reproduced by the fixture layout.
- PSTH: `evaluation.py:332 eval_psth` with explicit `psth`, `eval_rates`,
  `eval_cond_idx`, `jitter`.
- DMFC code: `evaluation.py:386 speed_tp_correlation`.
- DMFC global: no `nlb_tools` function; use a separate Python script that
  computes the same per-trial speed and a single `scipy.stats.pearsonr`.
- Velocity: `evaluation.py:437 velocity_decoding`, including the default
  `GridSearchCV` path.

The fixture environment is fixed for this plan at:

- Python 3.13.5
- numpy 2.5.4
- scipy 1.18.1
- scikit-learn 1.9.1

`nlb_tools` does not pin scikit-learn; this is lvm-eval’s chosen reference
environment. If a different version is used, update §12 and regenerate every
affected golden; do not mix versions.

### 5.2 Fixture layout

```text
test/fixtures/
  bits/        inputs/*.npy, expected.txt, meta.txt
  forward/     inputs/*.npy, expected.txt, meta.txt
  psth/        inputs/*.npy, expected.txt, meta.txt
  dmfc_code/   inputs/*.npy, expected.txt, meta.txt
  dmfc_global/ inputs/*.npy, expected.txt, meta.txt
  decode/      inputs/*.npy, expected.txt, meta.txt
```

Fixtures are plain `.npy` arrays of exactly what a caller passes plus a
plain-text expected scalar and metadata. Tests may use the `nx.io` sublibrary
to read fixtures; that is test-only and never part of the metric core.
`meta.txt` records shapes, dtypes, seed (if any), and generator versions.

Ragged trials are rectangularized before the call: pad the time axis with
NaN, and supply an explicit boolean mask where the metric takes one. Fixtures
contain no jagged/object arrays. Padded entries are never scored. Decoder
fixtures follow the reference order: select mask trials, flatten trial-major,
then drop flattened samples with NaN in the first behaviour column. PSTH
fixtures use NaN padding and the first-neuron NaN row selector.

Synthetic fixtures come first. Real-data fixtures are blocked (see §13).

### 5.3 Tolerance and precision

- **Tolerance decision:** relative tolerance `1e-6` in float64, with an
  absolute floor of `1e-12` for expected zero. Do not loosen without approval.
- Compare finite scalars only; a NaN/Inf expected value is a fixture error
  unless the case is explicitly testing an error path.
- Callers cast float32 inputs to float64 before the call; the metric core
  never accumulates bps/R²/Pearson sums in float32.
- `Nx.sum` order is unspecified; do not require bitwise equality.
- `Nx.lgamma` is documented within 16 ulps; use float64.
- If a Raven operation only gives float32, stop and ask before loosening the
  tolerance.

### 5.4 Tests

- Golden comparisons for every metric and variant.
- Shape/rank mismatch errors naming the function and argument.
- Mask equivalence: explicit mask versus equivalent NaN-spike input for bps.
- No mutation: inputs are byte-identical after every call, including zero-rate
  and mask cases.
- Edge cases in §9, including error paths.
- PSTH jitter sign and padding cases.
- Decode mask averaging: one column, multiple columns, empty mask.
- No file I/O/global state in metric modules.

Sanity checks that are valid by construction may be included: per-neuron mean
rates give exactly 0 bps when the null is computed the same way; a PSTH
compared with itself at zero jitter gives R² 1. Do not use “true behaviour as
a prediction gives decode R² 1” as a control; it is not a general identity.
Do not use distil’s `example/nlb_baseline.ml` as a golden reference; it is a
diagnostic with a different decoder protocol.

## 6. Pinning mechanism and user documentation

### 6.1 Pin

`lvm-eval` uses Dune package management, not opam, mirroring
`/home/fh397/projects/distil/dune-project:25-41`. The planned pin is:

```text
(pin
 (url "../murder")
 (package
  (name cmdargs)))
```

`cmdargs` is pinned only as the Dune recursive-pin anchor. `lvm-eval` does
not link or depend on `cmdargs`; it depends on Raven `nx`. A throwaway project
in `/tmp` confirmed that pinning `cmdargs` and depending only on `nx`/`rune`
resolves Raven at murder’s commit. The planned Dune package and public library are `lvm-eval`; the OCaml module is `Lvm_eval`; the internal Dune library name is `lvm_eval`; the license is ISC (matching murder and distil).
The package depends on:

```text
(ocaml (>= 5.5.0))
dune
nx
```

`nx.io` is test-only. No other numerics library is added.

The lock must resolve Raven at the commit murder pins:

- murder: `f2f0263ffc768f0adc49292b28d4e7fb6eda6069`
- Raven: `3984341140aef7bfc6f27eb14474f4a261a7b7bc`

Every Raven package entry that the lock does contain (at least
`dune.lock/nx.dev.pkg`, plus the Raven packages needed by `nx` and any test
dependency) must carry the Raven source URL above. The lock contains only the
packages `lvm-eval` needs and their transitive dependencies; it need not
contain every package murder pins. Pins are inherited; only needed packages
appear. Do not add a separate Raven pin.

### 6.2 Raven dev-package workaround

Dune 3.24 writes `(build (all_platforms ((dune))))` for pinned dev packages,
which runs a bare `dune build` in Raven’s monorepo; Raven’s hugin test stanzas
then fail outside a package context. After `dune pkg lock`, rewrite every
pinned dev `.pkg` file to the rule distil documents:

```text
(build
 (all_platforms
  ((action
    (run dune build -p %{pkg-self:name} -j %{jobs} %{pkg-self:name}.install)))))
```

This is the only permitted lock edit. Review the generated lock diff; do not
hand-edit it otherwise. The workaround is documented in
`/home/fh397/projects/distil/dune-project:12-23` and
`/home/fh397/projects/distil/README.md:105-114`.

### 6.3 User-facing documentation

`README.md` states that callers use Dune package management, that the planned
Dune package and public library are `lvm-eval`, the OCaml module is `Lvm_eval`, the internal library name is `lvm_eval`, and that the
license is ISC. It records that `lvm-eval` pins `../murder` and inherits Raven
at murder’s commit. `docs/nlb-protocol.md` records dataset values as
documentation only. No dataset configs or presets are compiled into the
library.

## 7. Milestones

The order is deliberate: shared helpers and bps first; forward is the same
core; PSTH; DMFC code; DMFC global; velocity decoding last. Each milestone is
one coherent task with a done-check.

### Milestone 1 — shared helpers + co-smoothing bps

Shared helpers first needed here:

- `Check`: rank/shape checks and clear `Invalid_argument` errors.
- `Mask`: valid-entry counts and masked sums/means.
- `Numeric`: float64 helpers and zero-rate floor on a copy.

Implement `Bits.co_smoothing` on top of these. Do not implement R², Pearson,
or ridge yet.

Done-check: a synthetic golden from `bits_per_spike` matches within 1e-6;
explicit-mask and NaN-spike fixtures agree; zero total spikes raises; zero-rate
input is not mutated; shape errors name the argument; `dune build @ci` and
`dune runtest` pass.

### Milestone 2 — forward-prediction wrapper

Reuse the `Bits` bps core. No new helper is needed.

Done-check: a synthetic forward fixture matches `bits_per_spike` within 1e-6;
all-neuron shape is required; masks work; no distil forward claim is made.

### Milestone 3 — PSTH R²

Shared helper first needed here: `R2` (per-output R² with the force-finite
rule). Implement `Psth.r2`; no PSTH construction.

Done-check: a fixture from `eval_psth` matches within 1e-6; positive/negative
jitter and NaN padding cases pass; empty conditions are skipped; all-empty
conditions raise; first-neuron-NaN assumption is tested; no mutation.

### Milestone 4 — DMFC_RSG code variant

Shared helper first needed here: `Pearson` (with explicit undefined handling).
Implement `Dmfc.speed_tp_code`.

Done-check: a fixture from `speed_tp_correlation` matches within 1e-6; the
mask and speed formula are tested on hand-built trials; per-condition
averaging matches; singleton/constant conditions raise; all-neuron rates are
required.

### Milestone 5 — DMFC_RSG global variant

Reuse `Pearson`. Implement `Dmfc.speed_tp_global`; no condition labels.

Done-check: a separate Python golden (same speeds, `scipy.stats.pearsonr`)
matches within 1e-6; same speed/mask as code variant; documented as a paper
reading, not an `nlb_tools` reproduction; undefined cases raise.

### Milestone 6 — velocity decoding

Shared helper first needed here: `Ridge` (centered solve, alpha grid,
contiguous K-fold, predict). Reuse `R2`. Implement
`Decode.velocity_decoding`.

Done-check: a fixture from `velocity_decoding` matches within 1e-6, including
the default 9-alpha/5-fold path; one and multiple decode-mask columns are
tested; no standardization; empty mask raises; training samples below `folds`
raises; NaN in a non-first behaviour column raises; constant behaviour uses
force-finite R²; no mutation.

## 8. Open questions

1. **Public dtype.** This plan uses `Nx.float64_t` at the public boundary and
   requires callers with float32 to cast first. This keeps one metric path and
   satisfies the float64 requirement. The approved report allowed float32
   inputs with an internal cast; confirm whether float64-only is preferred.

The scikit-learn/fixture environment is fixed in §5.1 and §12. GPU placement
is decided in §11. Real-data fixtures are blocked, not open; see §13.

## 9. Edge-case behaviour

The default for an undefined case is a clear `Invalid_argument` naming the
metric, the argument, the shape, and the failed precondition. Deliberate
deviations affect only degenerate inputs; for valid inputs the numeric result
is the reference result. A metric is never described as an exact reproduction
without the deviations in this table.

| Case | What `nlb_tools` does (cite) | What `lvm-eval` does | Classification |
|---|---|---|---|
| bps: zero total spikes | Divides by `np.nansum(spikes)` with no guard; result Inf/NaN (`evaluation.py:276`) | Raises “no scored spikes” | DELIBERATE DEVIATION FROM REFERENCE |
| bps: zero rate prediction | Replaces zeros with `1e-9` in place (`evaluation.py:241-246`) | Replaces zeros with `1e-9` on a copy; input unchanged | DELIBERATE DEVIATION FROM REFERENCE (mutation only; numeric result same for valid inputs) |
| bps: negative rate | Asserts `rates >= 0` (`evaluation.py:240`) | Clear `Invalid_argument` | REPRODUCES REFERENCE (rejects; clearer error) |
| bps: NaN spikes | Drops entries with NaN spikes; asserts no NaN rates remain (`evaluation.py:233-238`) | Same: explicit mask wins; otherwise valid = non-NaN spikes; NaN spike or rate on a valid entry raises | REPRODUCES REFERENCE |
| bps: zero-spike neuron | Null rate 0 is floored to `1e-9` (`evaluation.py:271-275`, `:241-246`) | Same | REPRODUCES REFERENCE |
| bps: negative spikes | No explicit check; `gammaln(spikes+1)` on negative values (`evaluation.py:248`) | Clear error: counts must be nonnegative | DELIBERATE DEVIATION FROM REFERENCE |
| bps: non-integer spikes | No check; `gammaln` accepts (`evaluation.py:248`) | Clear error: counts must be integer-valued | DELIBERATE DEVIATION FROM REFERENCE |
| bps: all-false mask | No mask argument; all-NaN spikes give NaN mean and zero denominator (`evaluation.py:271-276`) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| forward: zero total spikes | Same division as bps (`evaluation.py:276`) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| PSTH: condition with no eval trials | Skips it (`evaluation.py:367-368`) | Skips it | REPRODUCES REFERENCE |
| PSTH: all conditions empty | No explicit guard; `np.vstack([])` at `evaluation.py:381` | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| PSTH: no valid rows across all conditions | Empty stacked arrays reach `r2_score`, which raises `ValueError` (`evaluation.py:381-383`) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| PSTH: NaN prediction in a required row | Asserts (`evaluation.py:375-377`) | Clear error naming condition/time | REPRODUCES REFERENCE (rejects; clearer error) |
| PSTH: inconsistent first-neuron padding | Uses `psth[c,:,0]` only (`evaluation.py:373-374`); other-neuron NaNs are not masked | Same assumption; a non-finite result raises rather than returning NaN | REPRODUCES REFERENCE (documented first-neuron assumption) |
| PSTH: jitter length mismatch | `jitter[idx]` raises `IndexError` (`evaluation.py:370`) | Clear shape error | DELIBERATE DEVIATION FROM REFERENCE |
| DMFC code: singleton condition | scipy 1.18.1 `pearsonr` raises `ValueError` (“must have length at least 2”); `evaluation.py:430` | Clear error: condition needs at least two trials | DELIBERATE DEVIATION FROM REFERENCE (clear error instead of scipy exception) |
| DMFC code: constant speed or `tp` with at least two trials | scipy 1.18.1 `pearsonr` returns NaN; mean becomes NaN (`evaluation.py:430-432`) | Clear error: correlation undefined | DELIBERATE DEVIATION FROM REFERENCE |
| DMFC code: mask selects no time points for a trial | `compute_speed` returns NaN for the empty masked trial (`evaluation.py:412-415`); `pearsonr` then raises `ValueError` for a singleton condition and returns NaN for a condition with at least two trials (`evaluation.py:430-432`) | Clear error: mask must select at least one time point per scored trial | DELIBERATE DEVIATION FROM REFERENCE (clear error instead of NaN/scipy exception) |
| DMFC global: empty/singleton/constant | No `nlb_tools` implementation; the Python golden uses scipy 1.18.1 (`ValueError` for empty/singleton, NaN for constant) | Clear error | DELIBERATE DEVIATION FROM REFERENCE (no reference; Python golden) |
| velocity: empty decode mask | `Ridge.fit` on zero samples raises (`evaluation.py:481-488`) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| velocity: NaN in non-first behaviour column | After mask selection and trial-major flatten, drops samples only by column 0 (`evaluation.py:314-319`); Ridge then fails on NaN target | Selects mask trials, flattens trial-major, drops first-column NaN samples, then raises if any retained value is non-finite | DELIBERATE DEVIATION FROM REFERENCE (clear error instead of estimator failure) |
| velocity: all-NaN behaviour rows | Drops all rows; Ridge fails (`evaluation.py:314-319`) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| velocity: constant behaviour dimension | sklearn R² force-finite: 1 if perfect, otherwise 0 (`evaluation.py:329`) | Same force-finite rule | REPRODUCES REFERENCE |
| velocity: no decode mask | `evaluate` creates one all-True mask (`evaluation.py:125-133`) | Default one all-True column | REPRODUCES REFERENCE |
| velocity: training samples fewer than `folds` (default 5) | `GridSearchCV`/`KFold` raises (`evaluation.py:324-329`; sklearn default 5-fold) | Clear error | DELIBERATE DEVIATION FROM REFERENCE |
| bps: infinite rate | No explicit check; `rates - spikes*log(rates)` returns NaN (`evaluation.py:248`) | Clear `Invalid_argument`: rates must be finite | DELIBERATE DEVIATION FROM REFERENCE |

## 10. Decode masks and PSTH pooling

- **Velocity decoding fits one ridge per decode-mask column and averages the
  R² values.** This is the `nlb_tools` behaviour
  (`evaluation.py:479-491`), including the two Area2_Bump masks
  (`make_tensors.py:66`). It contradicts the old README’s “one linear map”
  wording; the README no longer says that. The default with no mask is one
  all-True column, matching `evaluate` (`evaluation.py:125-133`).
- **PSTH R² pools condition-time rows.** `eval_psth` stacks all valid rows
  from all conditions and computes one R² per neuron (`evaluation.py:381-383`),
  then sklearn averages over neurons. It is not an equal-weight average of
  per-condition R² values. Conditions with more valid time bins weigh more.
- **PSTH padding assumption.** Valid rows are selected by
  `~np.isnan(psth[c,:,0])` (`evaluation.py:373-374`): the first neuron defines
  the padding mask. `lvm-eval` reproduces that assumption; it does not invent a
  different all-neuron mask.

## 11. Execution model

- All metrics compute in float64 on the host CPU. The public API takes
  `Nx.float64_t`; callers with float32 cast once at the boundary. No metric
  accumulates sums in float32.
- Raven’s AMD backend does not implement `cholesky`, `svd`, or
  `solve_triangular` (Raven pin `3984341140aef7bfc6f27eb14474f4a261a7b7bc`,
  `packages/nx/lib/amd/backend.ml:1259-1266`). The library never calls those
  operations on an AMD device.
- Entry points require `Placement.host`; a non-host tensor raises
  `Invalid_argument` naming the tensor and its placement. There is no internal
  transfer.
- `Nx.sum` accumulation order is unspecified; use the tolerance in §5.3.
- `Nx.lgamma` is used for `lgamma(spikes+1)`; spike counts are nonnegative
  integers and the model/null `lgamma` terms cancel in bps but are retained to
  follow the reference arithmetic.
- No file I/O, global state, or model types in metric modules.

## 12. Versions

| Component | Version / commit |
|---|---|
| murder | `f2f0263ffc768f0adc49292b28d4e7fb6eda6069` |
| Raven | `3984341140aef7bfc6f27eb14474f4a261a7b7bc` |
| nlb_tools | `42f8410b88e12db136910fa2f888b025ea0aa2ae` |
| Dune package / library / public library / OCaml module / license | `lvm-eval` / `lvm-eval` / `Lvm_eval` / ISC |
| scikit-learn (fixture generator) | 1.9.1 (fixed; `nlb_tools` does not pin one) |
| numpy (fixture generator) | 2.5.4 |
| scipy (fixture generator) | 1.18.1 |
| Python (fixture generator) | 3.13.5 |

If a golden is regenerated with another scikit-learn/numpy/scipy version,
update this table and regenerate every affected fixture. Do not mix fixture
environments.

## 13. Real-data fixtures

`/home/fh397/projects/_refs/nlb_tools/data/eval_data_test.h5` is a 134-byte
Git LFS pointer, not the HDF5 target data. It was not fetched. Fixtures that
need real test targets are therefore blocked. Synthetic fixtures come first.

If real data is later authorized:

- fetch it through an approved process, outside the dune build;
- verify the license before committing any slice (NLB datasets are CC BY 4.0
  per paper §A.3.2; `nlb_tools` code is MIT);
- keep fixtures small and plain; commit only the exact arrays a caller would
  pass plus expected values;
- never add a dataset config or preset to the library.

## 14. Caller-side prerequisites (not lvm-eval work)

These are distil changes; `lvm-eval` never depends on distil.

| Metric | What distil must change |
|---|---|
| Co-smoothing | Create a held-in/held-out neuron split and retrain with held-in neurons only; convert rates from Hz to counts/bin (`rates * dt`); do not treat Bernoulli rates as NLB Poisson means without deciding the observation-model deviation; use the official split for NLB-comparable numbers. |
| Forward prediction | Provide an extra future window (reference: 200 ms) and all-neuron future spikes/rates. Current distil trials end at 700 ms, so forward evaluation is blocked until the data format changes. |
| PSTH R² | Build per-condition PSTHs from spikes at 1 ms, smooth with the symmetric Gaussian, then resample to 5 ms; map conditions; supply jitter; replace the current grand-mean PSTH. |
| DMFC_RSG speed–tp | No distil DMFC data today. If added, provide set-go mask, condition labels, `tp`, and all-neuron rates. |
| Velocity decoding | Align behaviour at the NLB lag (MC_Maze 100 ms; scaling 120 ms) or document a different alignment; pass per-bin rates; use the NLB decoder protocol (no standardization, sample-wise 5-fold) for NLB-comparable numbers; keep `Distil.Decode` only for distil diagnostics. |
| Dependency | Distil pins `lvm-eval`; `lvm-eval` never imports or depends on distil. The user plans to modify distil’s data format to the MC_Maze NLB format. |

## References

- Paper: `/home/fh397/projects/distil/_papers/NLB.pdf` §3.2, §3.3, §A.5.1,
  §A.5.2, §A.5.3, §A.6.
- Reference: `/home/fh397/projects/_refs/nlb_tools`, commit
  `42f8410b88e12db136910fa2f888b025ea0aa2ae`; `nlb_tools/evaluation.py`,
  `nlb_tools/make_tensors.py`, `nlb_tools/nwb_interface.py`.
- Dataset values: `docs/nlb-protocol.md`.
