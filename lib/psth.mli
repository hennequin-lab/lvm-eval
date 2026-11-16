(** PSTH R².

    The metric is defined in the NLB'21 paper, section 3.3, and its details in
    A.5.3 and A.6. The reference is
    [nlb_tools/evaluation.py:332 eval_psth]. The PSTH target is precomputed by
    the caller; this module does not smooth, resample, average or otherwise
    construct it. *)

(** [r2 ~rates_eval ~psth ~eval_cond_idx ?jitter ()] computes PSTH R².

    [rates_eval] is the held-out model output on eval trials, with shape
    [trial;time;neuron]. [psth] is the held-out target, with shape
    [condition;time;neuron], already smoothed and resampled by the caller and
    NaN-padded. [eval_cond_idx] maps each condition to trial indices in
    [rates_eval]. [jitter] is an optional integer-bin shift per eval trial;
    its length is the eval-trial count and it defaults to zeros.

    For each condition with eval trials, trials are shifted by their jitter,
    rate predictions are averaged within the condition, rows are selected by
    the first PSTH neuron's non-NaN mask, and the selected condition-time rows
    are pooled. R² is computed per neuron and averaged across neurons.

    Raises [Invalid_argument] on shape, rank, placement or index errors, on
    non-finite selected predictions or PSTH values, and when no valid rows
    remain. *)
val r2
  :  rates_eval:Nx.host Nx.float64_t
  -> psth:Nx.host Nx.float64_t
  -> eval_cond_idx:int array array
  -> ?jitter:int array
  -> unit
  -> float
