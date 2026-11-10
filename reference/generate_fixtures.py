"""Generate synthetic co-smoothing fixtures with the NLB reference.

This script is deliberately synthetic-only. It imports the pinned nlb_tools
checkout through PYTHONPATH and writes only under test/fixtures/bits and
test/fixtures/forward. It does not read distil data, LFS data, or any real
neural array.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np

from nlb_tools.evaluation import bits_per_spike, eval_psth

EXPECTED_NLB_TOOLS_COMMIT = "42f8410b88e12db136910fa2f888b025ea0aa2ae"
REFERENCE_ROOT = Path("/home/fh397/projects/_refs/nlb_tools")
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "test" / "fixtures" / "bits"
INPUTS = FIXTURE_ROOT / "inputs"
FORWARD_ROOT = REPO_ROOT / "test" / "fixtures" / "forward"
FORWARD_INPUTS = FORWARD_ROOT / "inputs"
PSTH_ROOT = REPO_ROOT / "test" / "fixtures" / "psth"
PSTH_INPUTS = PSTH_ROOT / "inputs"
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

# 2 trials x 3 future bins x 3 all neurons. All values are synthetic.
FORWARD_RATES = np.array(
    [
        [[0.9, 1.1, 0.4], [1.3, 0.6, 0.8], [0.5, 1.7, 0.2]],
        [[1.2, 0.8, 0.7], [0.4, 1.5, 1.1], [0.9, 0.5, 1.6]],
    ],
    dtype=np.float64,
)
FORWARD_SPIKES = np.array(
    [
        [[1.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 2.0, 1.0]],
        [[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]],
    ],
    dtype=np.float64,
)

# 3 eval trials x 5 time bins x 2 neurons and 2 conditions x 5 time bins x 2
# neurons. All values are synthetic.
PSTH_RATES = np.array(
    [
        [[0.8, 1.1], [1.2, 0.7], [0.5, 1.5], [1.4, 0.6], [0.9, 1.0]],
        [[1.3, 0.6], [0.7, 1.2], [1.1, 0.8], [0.4, 1.7], [1.6, 0.5]],
        [[0.6, 1.4], [1.5, 0.5], [0.3, 1.3], [1.2, 0.9], [0.8, 1.1]],
    ],
    dtype=np.float64,
)
PSTH_TARGET = np.array(
    [
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]],
        [[2.0, 1.0], [2.1, 1.1], [2.2, 1.2], [2.3, 1.3], [2.4, 1.4]],
    ],
    dtype=np.float64,
)
PSTH_COND_IDX = [np.array([0, 1], dtype=np.int64), np.array([2], dtype=np.int64)]
PSTH_JITTER = np.array([1, 1, -1], dtype=np.int64)


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
    "inf_rate",
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

    all_nan_neuron_spikes = NORMAL_SPIKES.copy()
    all_nan_neuron_spikes[:, :, 0] = np.nan
    cases.append(
        _case("all_nan_neuron", NORMAL_RATES.copy(), all_nan_neuron_spikes)
    )

    mean_rate = np.broadcast_to(
        np.mean(NORMAL_SPIKES, axis=(0, 1), keepdims=True), NORMAL_SPIKES.shape
    ).copy()
    cases.append(_case("mean_rate", mean_rate, NORMAL_SPIKES.copy()))

    cases.append(
        _case("zero_total_spikes", NORMAL_RATES.copy(), np.zeros_like(NORMAL_SPIKES))
    )

    negative_rate = NORMAL_RATES.copy()
    negative_rate[0, 0, 0] = -0.5
    cases.append(_case("negative_rate", negative_rate, NORMAL_SPIKES.copy()))

    negative_spikes = NORMAL_SPIKES.copy()
    negative_spikes[0, 0, 0] = -1.0
    cases.append(_case("negative_spikes", NORMAL_RATES.copy(), negative_spikes))

    inf_rate = NORMAL_RATES.copy()
    inf_rate[0, 0, 0] = np.inf
    cases.append(_case("inf_rate", inf_rate, NORMAL_SPIKES.copy()))

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


def build_forward_cases():
    cases = [
        _case("forward_normal", FORWARD_RATES.copy(), FORWARD_SPIKES.copy()),
    ]

    nan_spikes = FORWARD_SPIKES.copy()
    nan_spikes[0, 0, 0] = np.nan
    nan_spikes[1, 2, 2] = np.nan
    cases.append(
        _case(
            "forward_nan",
            FORWARD_RATES.copy(),
            nan_spikes,
            mask=~np.isnan(nan_spikes),
        )
    )

    return cases


def build_psth_cases():
    cases = []

    cases.append(
        {
            "name": "normal",
            "psth": PSTH_TARGET.copy(),
            "rates": PSTH_RATES.copy(),
            "cond_idx": [ids.copy() for ids in PSTH_COND_IDX],
            "jitter": None,
            "kind": "numeric",
        }
    )

    jitter_psth = PSTH_TARGET.copy()
    jitter_psth[0, 0, :] = np.nan
    jitter_psth[1, -1, :] = np.nan
    cases.append(
        {
            "name": "jitter",
            "psth": jitter_psth,
            "rates": PSTH_RATES.copy(),
            "cond_idx": [ids.copy() for ids in PSTH_COND_IDX],
            "jitter": PSTH_JITTER.copy(),
            "kind": "numeric",
        }
    )

    cases.append(
        {
            "name": "no_eval_condition",
            "psth": PSTH_TARGET.copy(),
            "rates": PSTH_RATES.copy(),
            "cond_idx": [np.array([0, 1], dtype=np.int64), np.array([], dtype=np.int64)],
            "jitter": None,
            "kind": "numeric",
        }
    )

    cases.append(
        {
            "name": "all_empty",
            "psth": PSTH_TARGET.copy(),
            "rates": PSTH_RATES.copy(),
            "cond_idx": [np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
            "jitter": None,
            "kind": "error",
        }
    )

    no_valid_psth = np.full_like(PSTH_TARGET, np.nan)
    cases.append(
        {
            "name": "no_valid_rows",
            "psth": no_valid_psth,
            "rates": PSTH_RATES.copy(),
            "cond_idx": [ids.copy() for ids in PSTH_COND_IDX],
            "jitter": None,
            "kind": "error",
        }
    )

    nan_rates = PSTH_RATES.copy()
    nan_rates[0, 0, 0] = np.nan
    cases.append(
        {
            "name": "nan_prediction",
            "psth": PSTH_TARGET.copy(),
            "rates": nan_rates,
            "cond_idx": [ids.copy() for ids in PSTH_COND_IDX],
            "jitter": None,
            "kind": "error",
        }
    )

    return cases


def _save_cond_idx(name, cond_idx):
    max_len = max((len(ids) for ids in cond_idx), default=0)
    padded = np.full((len(cond_idx), max_len), -1, dtype=np.int64)
    for i, ids in enumerate(cond_idx):
        padded[i, : len(ids)] = ids
    np.save(PSTH_INPUTS / f"{name}_cond_idx.npy", padded)
    np.save(
        PSTH_INPUTS / f"{name}_cond_sizes.npy",
        np.array([len(ids) for ids in cond_idx], dtype=np.int64),
    )


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
    common_meta = [
        "generator: reference/generate_fixtures.py",
        "fixture_kind: synthetic",
        f"seed: {SEED}",
        f"python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"scipy: {importlib.metadata.version('scipy')}",
        f"scikit-learn: {importlib.metadata.version('scikit-learn')}",
        f"h5py: {h5py.__version__}",
        f"nlb_tools_commit: {EXPECTED_NLB_TOOLS_COMMIT}",
        "reference_note: expected values are computed on copies of reference inputs where the reference mutates them in place",
        "",
    ]
    meta_lines = common_meta + []

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

    FORWARD_INPUTS.mkdir(parents=True, exist_ok=True)
    forward_expected = []
    forward_meta = common_meta + []
    for case in build_forward_cases():
        name = case["name"]
        rates = case["rates"]
        spikes = case["spikes"]
        mask = case["mask"]

        np.save(FORWARD_INPUTS / f"{name}_rates.npy", rates)
        np.save(FORWARD_INPUTS / f"{name}_spikes.npy", spikes)
        if mask is not None:
            np.save(FORWARD_INPUTS / f"{name}_mask.npy", mask)

        expected = float(bits_per_spike(rates.copy(), spikes.copy()))
        forward_expected.append(f"{name} {expected:.17g}")
        forward_meta.append(
            f"case {name}: rates shape={rates.shape} dtype={rates.dtype}; "
            f"spikes shape={spikes.shape} dtype={spikes.dtype}; "
            f"mask={'none' if mask is None else f'{mask.shape} {mask.dtype}'}; "
            f"expected=numeric"
        )

    (FORWARD_ROOT / "expected.txt").write_text("\n".join(forward_expected) + "\n")
    (FORWARD_ROOT / "meta.txt").write_text("\n".join(forward_meta) + "\n")

    PSTH_INPUTS.mkdir(parents=True, exist_ok=True)
    psth_expected = []
    psth_meta = common_meta + []
    for case in build_psth_cases():
        name = case["name"]
        psth = case["psth"]
        rates = case["rates"]
        cond_idx = case["cond_idx"]
        jitter = case["jitter"]

        np.save(PSTH_INPUTS / f"{name}_psth.npy", psth)
        np.save(PSTH_INPUTS / f"{name}_rates.npy", rates)
        _save_cond_idx(name, cond_idx)
        if jitter is not None:
            np.save(PSTH_INPUTS / f"{name}_jitter.npy", jitter)

        if case["kind"] == "numeric":
            expected = float(
                eval_psth(
                    psth.copy(),
                    rates.copy(),
                    [ids.copy() for ids in cond_idx],
                    jitter=None if jitter is None else jitter.copy(),
                )
            )
            psth_expected.append(f"{name} {expected:.17g}")
        else:
            psth_expected.append(f"{name} error")

        psth_meta.append(
            f"case {name}: psth shape={psth.shape} dtype={psth.dtype}; "
            f"rates shape={rates.shape} dtype={rates.dtype}; "
            f"conditions={len(cond_idx)}; jitter={'none' if jitter is None else 'integer bins'}; "
            f"expected={'numeric' if case['kind'] == 'numeric' else 'error'}"
        )

    (PSTH_ROOT / "expected.txt").write_text("\n".join(psth_expected) + "\n")
    (PSTH_ROOT / "meta.txt").write_text("\n".join(psth_meta) + "\n")

    status_after = subprocess.check_output(
        ["git", "-C", str(REFERENCE_ROOT), "status", "--porcelain"], text=True
    )
    if status_after.strip():
        raise SystemExit(f"nlb_tools changed during generation:\n{status_after}")

    print(
        f"wrote {len(build_cases())} bits cases to {FIXTURE_ROOT} and "
        f"{len(build_forward_cases())} forward cases to {FORWARD_ROOT} and "
        f"{len(build_psth_cases())} psth cases to {PSTH_ROOT}"
    )


if __name__ == "__main__":
    main()
