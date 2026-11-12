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

Use Dune package management. The Dune package and public library are
`lvm-eval`; the OCaml module is `Lvm_eval`; the internal Dune library name is
`lvm_eval`; the license is ISC. A caller pins the public repository at a
release tag or full commit; `lvm-eval` pins
`https://github.com/hennequin-lab/murder` at commit
`f2f0263ffc768f0adc49292b28d4e7fb6eda6069` and inherits Raven at murder's
commit. No sibling checkout is required. The exact pin, lock and Raven
dev-package workaround are in [`plan.md`](plan.md) §6.

```text
(pin
 (url
  "git+https://github.com/hennequin-lab/lvm-eval.git#e5fbbce0dbe014f2312cb616647bfdf8410aa5c9")
 (package (name lvm-eval)))
; caller's dune file
(libraries lvm-eval)
```

After the caller's own `dune pkg lock`, apply the Raven dev-package workaround
from [`plan.md`](plan.md) §6.2 to the caller's lock. Raven is a monorepo whose
package-private tests otherwise fail the pinned-dev build; the workaround is
required for a clean consumer build.

`lvm-eval` never depends on its callers.

## Status

First release implemented. The Dune project, library, tests, and synthetic
golden fixtures are in place. There is no CLI and no file I/O in the
first release. [`plan.md`](plan.md) is the implementation plan and review record,
including the public API, metric specifications, verification strategy,
milestones, edge-case behaviour, and known deviations.
[`docs/nlb-protocol.md`](docs/nlb-protocol.md) records the dataset-level NLB
values for callers; those values are documentation, not code presets.

The first release reproduces the `nlb_tools` NLB metrics for valid inputs.
The DMFC_RSG global variant is a documented paper reading, not an
`nlb_tools` reproduction. Deliberate deviations are limited to degenerate
inputs and are listed in `plan.md` §9.

## License

ISC.

## References

- Pei et al., *Neural Latents Benchmark '21: Evaluating latent variable models
  of neural population activity*, NeurIPS 2021 Datasets and Benchmarks. Paper:
  `https://arxiv.org/html/2109.04463v4`.
- Reference implementation: `https://github.com/neurallatents/nlb_tools`, commit
  `42f8410b88e12db136910fa2f888b025ea0aa2ae`.
