(** Velocity decoding R².

    The metric is defined in the NLB'21 paper, section 3.3, with decoding lags
    in A.5.2. The reference is
    [nlb_tools/evaluation.py:279 fit_and_eval_decoder] and
    [nlb_tools/evaluation.py:437 velocity_decoding]. The lag is caller data
    preparation; this module assumes rates and behaviour are already aligned. *)

(** [velocity_decoding ~rates_train ~behaviour_train ~rates_eval ~behaviour_eval ?train_decode_mask ?eval_decode_mask ?alphas ?folds ()]
    computes velocity-decoding R².

    Rates are [trial;time;neuron]; behaviour is [trial;time;dim]. Decode masks
    are [trial;mask] bool and must both be present or both absent. Each mask
    column selects trials, then the selected trial-time samples are flattened
    trial-major. Samples with NaN in the first behaviour column are dropped,
    and all retained rates/behaviour must be finite.

    One centered ridge is fitted per mask column, with an unpenalized
    intercept, using the reference alpha grid and contiguous K-fold selection
    over flattened samples. The default grid is [Ridge.default_alphas] and the
    default is 5 folds. R² is computed per behaviour dimension and averaged,
    then averaged across mask columns.

    There is no feature standardization and no decoding lag inside this
    function. *)
val velocity_decoding
  :  rates_train:Nx.float64_t
  -> behaviour_train:Nx.float64_t
  -> rates_eval:Nx.float64_t
  -> behaviour_eval:Nx.float64_t
  -> ?train_decode_mask:Nx.bool_t
  -> ?eval_decode_mask:Nx.bool_t
  -> ?alphas:float array
  -> ?folds:int
  -> unit
  -> float
