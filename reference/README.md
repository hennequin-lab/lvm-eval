# Reference fixture generator

`generate_fixtures.py` generates the synthetic co-smoothing fixtures under
`test/fixtures/bits/`, the synthetic forward-prediction fixtures under
`test/fixtures/forward/`, the synthetic PSTH fixtures under
`test/fixtures/psth/`, and the synthetic DMFC code-variant fixtures under
`test/fixtures/dmfc_code/`. It imports the pinned `nlb_tools` checkout read-only
through `PYTHONPATH` and does not install into it or write inside it. It
imports `nlb_tools.evaluation.bits_per_spike`,
`nlb_tools.evaluation.eval_psth`, and
`nlb_tools.evaluation.speed_tp_correlation`; no pandas or pynwb is needed for
these metric paths.

The reference checkout is:

- path: `/home/fh397/projects/_refs/nlb_tools`
- commit: `42f8410b88e12db136910fa2f888b025ea0aa2ae`

## Environment

- Python `3.13.5`
- numpy `2.5.4`
- scipy `1.18.1`
- scikit-learn `1.9.1`
- h5py `3.16.0`

Venv used: `/tmp/nlb-venv`. `nlb_tools` was not installed, and
`requirements.txt` was not used.

```text
cloudpickle==3.1.2
h5py==3.16.0
joblib==1.6.0
narwhals==2.28.0
numpy==2.5.4
scikit-learn==1.9.1
scipy==1.18.1
threadpoolctl==3.8.0
```

## Generate

From the lvm-eval repository root:

```sh
PYTHONPATH=/home/fh397/projects/_refs/nlb_tools \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/nlb-venv/bin/python reference/generate_fixtures.py
```

The expected bits/spike is computed with `nlb_tools.evaluation.bits_per_spike`
on a copy of the rates because the reference mutates zero rates in place. The
fixture input is the original rates array. Outputs are synthetic only under
`test/fixtures/bits/`, `test/fixtures/forward/`, `test/fixtures/psth/`, and
`test/fixtures/dmfc_code/`; no real data, LFS data, or distil data is used.
PSTH expected values come from `nlb_tools.evaluation.eval_psth`; PSTH
construction is not part of this generator. DMFC code-variant expected values
come from `nlb_tools.evaluation.speed_tp_correlation`.

Before running, verify the reference checkout is clean and at the commit above:

```sh
git -C /home/fh397/projects/_refs/nlb_tools status --porcelain
git -C /home/fh397/projects/_refs/nlb_tools rev-parse HEAD
```
