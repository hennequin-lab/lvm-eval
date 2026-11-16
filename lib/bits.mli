(** Co-smoothing bits/spike.

    The metric is defined in the NLB'21 paper, section 3.2, displayed bits/spike
    equation. The reference implementation is
    [nlb_tools/evaluation.py:209 neg_log_likelihood] and
    [nlb_tools/evaluation.py:252 bits_per_spike]. *)

(** [co_smoothing ~rates_heldout ~spikes_heldout ?mask_heldout ()] computes
    co-smoothing bits/spike for a [trial;time;neuron] tensor pair.

    [rates_heldout] is the model output for held-out neurons: expected spike
    counts per bin (Poisson means), not Hz. [spikes_heldout] is the held-out
    target: observed spike counts per bin. [mask_heldout] optionally marks the
    scored entries; when absent, non-NaN [spikes_heldout] entries are scored.

    The metric uses the global scored spike total as [n_sp], a per-neuron null
    mean over the scored entries, a [1e-9] floor for exact zero rates on a copy,
    and [log 2] normalization. It raises [Invalid_argument] on shape or
    placement errors and on undefined inputs. *)
val co_smoothing
  :  rates_heldout:Nx.host Nx.float64_t
  -> spikes_heldout:Nx.host Nx.float64_t
  -> ?mask_heldout:Nx.host Nx.bool_t
  -> unit
  -> float

(** [forward_prediction ~rates_forward ~spikes_forward ?mask_forward ()]
    computes forward-prediction bits/spike for future
    [trial;forward_time;all_neuron] tensors.

    The metric is defined in the NLB'21 paper, section 3.3. The reference is
    [nlb_tools/evaluation.py:14 evaluate] (forward branch and its
    [np.dstack] of held-in and held-out neurons) and
    [nlb_tools/evaluation.py:252 bits_per_spike]. The future-window length is
    caller data preparation; this function has no window parameter.

    [rates_forward] is the model output for future bins across all neurons:
    expected spike counts per bin, not Hz. [spikes_forward] is the held-out
    future target. [mask_forward] optionally marks the scored entries; when
    absent, non-NaN [spikes_forward] entries are scored. The arithmetic is the
    same bps core as {!co_smoothing}. *)
val forward_prediction
  :  rates_forward:Nx.host Nx.float64_t
  -> spikes_forward:Nx.host Nx.float64_t
  -> ?mask_forward:Nx.host Nx.bool_t
  -> unit
  -> float
