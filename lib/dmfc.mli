(** DMFC_RSG speed–`tp` correlation.

    The metric is defined in the NLB'21 paper, section 3.3, with the set-go
    window and jitter details in A.6. The reference is
    [nlb_tools/evaluation.py:386 speed_tp_correlation]. This module currently
    implements only the reference per-condition code variant. *)

(** [speed_tp_code ~rates_eval ~mask ~conditions ~tp ()] computes the
    per-condition Pearson r between per-trial neural speed and [tp], then
    averages across conditions.

    [rates_eval] is [trial;time;neuron]. [mask] is [trial;time] and marks the
    set-go time points. [conditions] is one integer label per trial; for the
    reference, encode the tuple `(is_eye, theta, is_short)`. [tp] is the
    produced interval in seconds.

    Neural speed is the mean over selected time points of the L2 norm across
    neurons of the time difference of rates. It raises [Invalid_argument] on
    shape or index errors, a trial with no selected time point, fewer than two
    trials in a condition, constant speed or [tp], or non-finite values. *)
val speed_tp_code
  :  rates_eval:Nx.float64_t
  -> mask:Nx.bool_t
  -> conditions:int array
  -> tp:float array
  -> unit
  -> float

(** [speed_tp_global ~rates_eval ~mask ~tp ()] computes one Pearson r across
    all eval trials between per-trial neural speed and [tp].

    This is a documented minimal reading of NLB.pdf §3.3, not an [nlb_tools]
    implementation. It uses the same per-trial speed and set-go mask as
    {!speed_tp_code}, but ignores condition labels. *)
val speed_tp_global
  :  rates_eval:Nx.float64_t
  -> mask:Nx.bool_t
  -> tp:float array
  -> unit
  -> float
