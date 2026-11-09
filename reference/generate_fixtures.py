"""Generate synthetic co-smoothing fixtures with the NLB reference.

This script is deliberately synthetic-only. It imports the pinned nlb_tools
checkout through PYTHONPATH and writes only under test/fixtures/bits. It does
not read distil data, LFS data, or any real neural array.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np

from nlb_tools.evaluation import bits_per_spike

EXPECTED_NLB_TOOLS_COMMIT = "42f8410b88e12db136910fa2f888b025ea0aa2ae"
REFERENCE_ROOT = Path("/home/fh397/projects/_refs/nlb_tools")
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "test" / "fixtures" / "bits"
INPUTS = FIXTURE_ROOT / "inputs"
SEED = 0

# 2 trials x 3 time bins x 2 held-out neurons. All values are synthetic.
NORMAL_RATES = np.array(
    [
        [[0.8, 1.2], [1.6, 0.4], [0.2, 2.0]],
        [[1.1, 0.7], [0.3, 1.9], [1.4, 0.6]],
    ],
    dtype=np.float64,
)
NORMAL_SPIKES = np.array(
    [
        [[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]],
        [[1.0, 1.0], [0.0, 2.0], [2.0, 0.0]],
    ],
    dtype=np.float64,
)


def _case(name: str, rates: np.ndarray, spikes: np.ndarray, *, mask: np.ndarray | None = None):
    return {
        "name": name,
        "rates": rates,
        "spikes": spikes,
        "mask": mask,
        "kind": "numeric" if name not in ERROR_CASES else "error",
    }


ERROR_CASES = {
    "zero_total_spikes",
    "negative_rate",
    "negative_spikes",
    "non_integer_spikes",
    "all_false_mask",
}


def build_cases():
    cases = []

    cases.append(_case("normal", NORMAL_RATES.copy(), NORMAL_SPIKES.copy()))

    zero_rate_rates = NORMAL_RATES.copy()
    zero_rate_rates[0, 0, 0] = 0.0
    cases.append(_case("zero_rate", zero_rate_rates, NORMAL_SPIKES.copy()))

    nan_spikes = NORMAL_SPIKES.copy()
    nan_spikes[0, 0, 0] = np.nan
    nan_spikes[1, 2, 1] = np.nan
    nan_mask = ~np.isnan(nan_spikes)
    cases.append(_case("nan_spikes", NORMAL_RATES.copy(), nan_spikes, mask=nan_mask))

    zero_neuron_spikes = NORMAL_SPIKES.copy()
    zero_neuron_spikes[:, :, 1] = 0.0
    cases.append(
        _case("zero_spike_neuron", NORMAL_RATES.copy(), zero_neuron_spikes)
    )

    cases.append(
        _case("zero_total_spikes", NORMAL_RATES.copy(), np.zeros_like(NORMAL_SPIKES))
    )

    negative_rate = NORMAL_RATES.copy()
    negative_rate[0, 0, 0] = -0.5
    cases.append(_case("negative_rate", negative_rate, NORMAL_SPIKES.copy()))

    negative_spikes = NORMAL_SPIKES.copy()
    negative_spikes[0, 0, 0] = -1.0
    cases.append(_case("negative_spikes", NORMAL_RATES.copy(), negative_spikes))

    non_integer_spikes = NORMAL_SPIKES.copy()
    non_integer_spikes[0, 0, 0] = 1.5
    cases.append(
        _case("non_integer_spikes", NORMAL_RATES.copy(), non_integer_spikes)
    )

    false_mask = np.zeros(NORMAL_SPIKES.shape, dtype=bool)
    cases.append(
        _case("all_false_mask", NORMAL_RATES.copy(), NORMAL_SPIKES.copy(), mask=false_mask)
    )

    return cases


def main() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_NLB_TOOLS_COMMIT:
        raise SystemExit(f"nlb_tools HEAD is {head}, expected {EXPECTED_NLB_TOOLS_COMMIT}")
    status = subprocess.check_output(
        ["git", "-C", str(REFERENCE_ROOT), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise SystemExit(f"nlb_tools is not clean:\n{status}")

    rng = np.random.default_rng(SEED)
    # The reference has no random path for bps; draw once so the seed is real
    # and record that bps itself does not consume randomness.
    _ = rng.random(1)

    INPUTS.mkdir(parents=True, exist_ok=True)
    expected_lines = []
    meta_lines = [
        "generator: reference/generate_fixtures.py",
        "fixture_kind: synthetic",
        f"seed: {SEED}",
        f"python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"scipy: {importlib.metadata.version('scipy')}",
        f"scikit-learn: {importlib.metadata.version('scikit-learn')}",
        f"h5py: {h5py.__version__}",
        f"nlb_tools_commit: {EXPECTED_NLB_TOOLS_COMMIT}",
        "reference_note: expected bps are computed on a copy of rates because neg_log_likelihood mutates zero rates in place",
        "",
    ]

    for case in build_cases():
        name = case["name"]
        rates = case["rates"]
        spikes = case["spikes"]
        mask = case["mask"]

        np.save(INPUTS / f"{name}_rates.npy", rates)
        np.save(INPUTS / f"{name}_spikes.npy", spikes)
        if mask is not None:
            np.save(INPUTS / f"{name}_mask.npy", mask)

        if case["kind"] == "numeric":
            expected = float(bits_per_spike(rates.copy(), spikes.copy()))
            expected_lines.append(f"{name} {expected:.17g}")
        else:
            expected_lines.append(f"{name} error")

        meta_lines.append(
            f"case {name}: rates shape={rates.shape} dtype={rates.dtype}; "
            f"spikes shape={spikes.shape} dtype={spikes.dtype}; "
            f"mask={'none' if mask is None else f'{mask.shape} {mask.dtype}'}; "
            f"expected={'numeric' if case['kind'] == 'numeric' else 'error'}"
        )

    (FIXTURE_ROOT / "expected.txt").write_text("\n".join(expected_lines) + "\n")
    (FIXTURE_ROOT / "meta.txt").write_text("\n".join(meta_lines) + "\n")

    print(f"wrote {len(build_cases())} cases to {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
