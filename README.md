# lvm-eval

Evaluation metrics for latent variable models (LVMs) of neural population activity, computed from inferred firing rates.

> **Status:** early design. There is no implementation yet. This README describes the intended scope, the first metric set, and the boundaries the library should keep. The API and module layout are proposals and will change. This repository currently contains only this README and AGENTS. There is no Dune project, library, test suite, or CLI yet. Everything under "Planned layout" and "Usage" is a design proposal, not an existing API.

**What this is / is not:** An OCaml metrics library; not a training framework, model zoo, or leaderboard.

## Why

LVMs are judged by what they predict, not by how their latents happen to be parameterised. The Neural Latents Benchmark (NLB'21) made this explicit: a model submits inferred rates, and the evaluator holds the spikes, behaviour, conditions, splits, and scoring protocol. That boundary is what makes the benchmark model-agnostic, and it is the boundary `lvm-eval` is built around.

The first goal is the four NLB metrics. The library is deliberately structured so that further rate-based metrics, and later metrics on other model outputs, can be added without changing the core.

## Metrics

### Co-smoothing (primary)

Poisson log-likelihood of held-out neurons' spike counts given predicted rates, normalised to bits per spike.

For held-out spike counts `y[n,t]` and predicted per-bin rates `λ[n,t]`, the Poisson log-likelihood up to the constant `log y!` is

```text
L(λ; y) = Σ (y log λ - λ)
```

and the score is

```text
bps = (L(λ; y) - L(λ̄; y)) / (n_sp log 2)
```

where `λ̄[n]` is the per-neuron mean spike count per bin over the scored entries and `n_sp` is the total number of scored spikes. A positive value means the model predicts the held-out activity better than a flat per-neuron mean rate.

The metric scores held-out rates; it is the model's job to condition on held-in neurons. `λ` is the Poisson mean **per bin**, not a rate in Hz. If a model emits Hz, it must be multiplied by the bin width before scoring.

### Forward prediction

The same Poisson bits/spike score as co-smoothing, with a different mask: future time bins across all neurons rather than held-out neurons. The likelihood and normalisation are identical, so both metrics share one implementation.

### Behaviour decoding

Ridge regression from training rates to behaviour, evaluated by R² on held-out trials. One linear map, no nonlinear decoder and no tap bank: the point is to test the rates, not to let a flexible decoder compensate for them. The decoding lag and the ridge-selection rule are protocol parameters, not hidden defaults.

### PSTH R²

Trial-averaged predicted rates within each condition compared to an empirical peri-stimulus time histogram (PSTH) built from spike counts. R² is computed per neuron across conditions and time, then averaged across neurons. The PSTH kernel width, trial exclusions, and per-trial jitter are protocol parameters.

## Design principles

- **Model-agnostic.** The evaluator knows rates, spikes, behaviour, conditions, masks, and a protocol. It knows nothing about latents, checkpoints, optimisers, or training code.
- **Explicit protocol.** Bin width, trial alignment, held-in/held-out splits, decoding lag, PSTH kernel, and ridge selection are data. Results carry the protocol that produced them.
- **Explicit observation model.** NLB is Poisson on spike counts. Other observation models are possible, but they are not NLB-comparable and must be named as such.
- **One likelihood, two masks.** Co-smoothing and forward prediction are the same `bits_per_spike` function.
- **No silent transformations.** A metric does not smooth, resample, or fit a readout unless that is part of the metric's definition. When the evaluator fits a readout, the result says so.
- **Float64 accumulation.** Log-likelihoods and R² are accumulated in double precision even when the tensors are float32.
- **Breakdowns, not just scalars.** Results include per-neuron and per-trial values alongside the headline number, so a score can be inspected rather than trusted.

## Data model (planned)

Tensors use `[time; trial; neuron]` throughout.

| Value | Shape | Meaning |
| --- | --- | --- |
| `rates` | `[time; trial; neuron]` | Expected spike count per bin (Poisson mean), not Hz |
| `spikes` | `[time; trial; neuron]` | Observed spike counts per bin |
| `mask` | `[time; trial; neuron]` | Entries that are scored; everything else is ignored |
| `behaviour` | `[time; trial; dim]` | Optional behavioural targets |
| `conditions` | `[trial]` | Optional condition label per trial |
| `protocol` | record | Bin width, splits, lag, kernel, folds, observation model |

Ragged trials are expected to be represented by padding plus an explicit mask. Trial alignment and per-trial jitter belong to the protocol, not to the metric.

## Planned layout

```text
lib/
  bits.ml       Poisson bits/spike core
  psth.ml       PSTH construction and R-squared
  decode.ml     Ridge regression and R-squared
  protocol.ml   Splits, masks, alignment, NLB presets
  result.ml     Metric values and per-unit breakdowns
test/
  test_bits.ml
  test_psth.ml
  test_decode.ml
  fixtures/     Small tensors and expected values
```

The library will be built on the Raven stack (`nx` for tensors). Data I/O and command-line tooling are expected to live outside the core so that the metrics stay pure and easy to test.

Later layers may add:

- `Latent` — probes and diagnostics that take latents rather than rates. These need an explicit invariance or readout contract, because latent coordinates are not identifiable across models.
- `Output` — metrics for other model outputs (behaviour predictions, continuous signals, and so on).

Neither is part of the first release.

## Comparability with NLB

To reproduce published NLB numbers, all of the following must match the benchmark:

| Parameter | NLB |
| --- | --- |
| Bin width | 5 ms |
| Co-smoothing observation | Poisson on held-out spike counts |
| Null model | Per-neuron mean spike count per bin |
| Behaviour decoder | Ridge, one linear map |
| Decoding lag | MC_Maze 100 ms, MC_RTT 140 ms, Area2_Bump −20 ms |
| PSTH kernel σ | MC_Maze 70 ms, Area2_Bump 40 ms, DMFC_RSG 70 ms |

If any of these differ, the result should be labelled **NLB-style**, not NLB. The split matters as much as the metric: scores on a non-official split are not comparable to the published table even when the code is identical.

## Validation plan

- Golden tests against `nlb_tools.evaluation` on small synthetic fixtures and one real NLB dataset. Same rates, spikes, masks, PSTH, and decoder should produce the same numbers to floating-point tolerance.
- No-model controls: per-neuron mean rates give exactly 0 bps; the empirical PSTH as a prediction gives R² 1; true behaviour as a prediction gives decode R² 1.
- Edge cases: zero-spike neurons, constant PSTHs, single-trial conditions, NaN padding, and jitter.
- A regression against the NLB-style spike-smoothing baseline in `distil`, to confirm that extracting the metrics did not change the numbers.

## Roadmap

- [ ] Poisson bits/spike core with explicit masks
- [ ] Co-smoothing and forward-prediction wrappers
- [ ] PSTH construction and R²
- [ ] Ridge behaviour decoding with protocol-selected lag and regularisation
- [ ] Protocol record and serialisation; NLB presets
- [ ] Result records with per-neuron and per-trial breakdowns
- [ ] Golden tests against `nlb_tools.evaluation`
- [ ] Simple tensor/manifest I/O and a thin CLI
- [ ] Additional rate-based metrics
- [ ] Optional non-rate outputs and latent probes

## Non-goals

- This is not the official NLB evaluator, and it is not a leaderboard.
- It is not a model zoo, training framework, or checkpoint format.
- It does not claim that latent coordinates are comparable across models. Any latent metric must be invariant to the allowed symmetry or must fit an explicit readout.
- It does not replace the official NLB split when official comparability is required.

## Usage

Not implemented yet. The intended shape is a small set of pure functions composed through a protocol, for example:

```ocaml
(* Illustrative; this API does not exist yet. *)
let protocol = Lvm_eval.Protocol.nlb `Mc_maze in
let result =
  Lvm_eval.Rate.evaluate
    ~protocol
    ~rates
    ~spikes
    ~behaviour
    ~conditions
```

## A note on the name

Here **LVM** means *latent variable model*, as in the NLB paper. It does not mean a large vision model or a Linux logical volume.

## Contributing

The project is at the design stage. Please open an issue before sending a pull request so the API can settle first. Once code lands, the checks will be `dune build @ci`, `dune runtest`, and `dune fmt`.

## License

TBD.

## References

- Pei et al., *Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity*, arXiv:[2109.04463](https://arxiv.org/abs/2109.04463).
- [`nlb_tools`](https://github.com/neurallatents/nlb_tools), the reference implementation and data tooling.
- The NLB datasets are hosted on [DANDI](https://dandiarchive.org/) and released under CC BY 4.0.
