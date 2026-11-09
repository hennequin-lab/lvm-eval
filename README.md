# lvm-eval

Evaluation metrics for latent variable models (LVMs) of neural population
activity, computed from model outputs and ground-truth targets.

`lvm-eval` is a pure OCaml library. It receives tensors and returns scalar
metric values. Tensors use `[trial; time; neuron]` throughout; rates are
expected spike counts per bin (Poisson means), not Hz. The first release
targets the NLB'21 metric set: co-smoothing bits/spike, forward-prediction
bits/spike, PSTH R², the DMFC_RSG speed–`tp` correlation (the `nlb_tools`
per-condition variant and a documented global variant), and velocity-decoding
R². All headline outputs are scalars.

## What it never contains

- models, weights, checkpoints, optimisers, or training code;
- data loading, trial splitting, trial alignment, binning, storage, or dataset
  configs;
- a dependency on `distil` or any other caller;
- file I/O or a CLI in the first release.

Callers own the data and the model. They pass held-in-derived model outputs,
held-out model outputs, and held-out targets as separate, clearly named
arguments. The library checks shape compatibility only; it does not validate
trial or neuron alignment.

## How to depend on it

Use Dune package management. The planned package/library name is `lvm_eval`.
A caller pins this repository; `lvm-eval` pins `../murder` itself and inherits
Raven at the commit murder pins. The exact pin, lock and Raven dev-package
workaround are in [`plan.md`](plan.md) §6.

```text
; planned; the Dune project does not exist yet
(pin (url "../lvm-eval") (package (name lvm_eval)))
```

`lvm-eval` never depends on its callers.

## Status

Early design. There is no Dune project, library, test suite, or CLI yet.
[`plan.md`](plan.md) is the implementation plan, including the public API
sketch, metric specifications, verification strategy, milestones, edge-case
behaviour, and known deviations. [`docs/nlb-protocol.md`](docs/nlb-protocol.md)
records the dataset-level NLB values for callers; those values are
documentation, not code presets.

The first release reproduces the `nlb_tools` NLB metrics for valid inputs.
The DMFC_RSG global variant is a documented paper reading, not an
`nlb_tools` reproduction. Deliberate deviations are limited to degenerate
inputs and are listed in `plan.md` §9.

## References

- Pei et al., *Neural Latents Benchmark '21: Evaluating latent variable models
  of neural population activity*, NeurIPS 2021 Datasets and Benchmarks. Paper:
  `/home/fh397/projects/distil/_papers/NLB.pdf`.
- Reference implementation: `/home/fh397/projects/_refs/nlb_tools`, commit
  `42f8410b88e12db136910fa2f888b025ea0aa2ae`.
