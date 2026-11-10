let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let co_smoothing ~rates_heldout ~spikes_heldout ?mask_heldout () =
  let fn = "Lvm_eval.Bits.co_smoothing" in
  Check.require_host fn "rates_heldout" rates_heldout;
  Check.require_host fn "spikes_heldout" spikes_heldout;
  Option.iter (Check.require_host fn "mask_heldout") mask_heldout;
  Check.require_rank3 fn "rates_heldout" rates_heldout;
  Check.require_rank3 fn "spikes_heldout" spikes_heldout;
  Check.require_same_shape
    fn
    "rates_heldout"
    "spikes_heldout"
    rates_heldout
    spikes_heldout;
  Option.iter
    (fun mask ->
       Check.require_same_shape fn "mask_heldout" "spikes_heldout" mask spikes_heldout)
    mask_heldout;
  let valid = Mask.valid ~fn ?mask:mask_heldout spikes_heldout in
  let counts = Mask.count_per_neuron valid in
  let spikes_valid = Mask.selected valid spikes_heldout in
  let rates_valid = Mask.selected valid rates_heldout in
  let all_finite_spikes = Nx.item [] (Nx.all (Nx.isfinite spikes_valid)) in
  if not all_finite_spikes then invalid fn "spikes_heldout contains NaN on a scored entry";
  let all_finite_rates = Nx.item [] (Nx.all (Nx.isfinite rates_valid)) in
  if not all_finite_rates
  then invalid fn "rates_heldout contains NaN or infinity on a scored entry";
  if Nx.item [] (Nx.any (Nx.less_s spikes_valid 0.))
  then invalid fn "spikes_heldout must be nonnegative";
  if Nx.item [] (Nx.any (Nx.less_s rates_valid 0.))
  then invalid fn "rates_heldout must be nonnegative";
  if not (Nx.item [] (Nx.all (Nx.equal spikes_valid (Nx.floor spikes_valid))))
  then invalid fn "spikes_heldout must be integer-valued";
  let n_sp = Mask.total valid spikes_heldout in
  if Nx.item [] (Nx.less_equal_s n_sp 0.)
  then invalid fn "no scored spikes (zero total spikes)";
  let rates_floor = Numeric.floor_zero_rates rates_valid in
  let safe_counts =
    Nx.where (Nx.greater counts (Nx.scalar Nx.float64 0.)) counts (Nx.ones_like counts)
  in
  let null_rates = Nx.Infix.(Mask.sum_per_neuron valid spikes_heldout / safe_counts) in
  let null_rates = Numeric.floor_zero_rates null_rates in
  let n = (Nx.shape spikes_heldout).(2) in
  let null_rates =
    null_rates |> Nx.reshape [| 1; 1; n |] |> Nx.broadcast_to (Nx.shape spikes_heldout)
  in
  let nll rates =
    let term =
      Nx.Infix.(rates - (spikes_valid * Nx.log rates) + Nx.lgamma (spikes_valid +$ 1.))
    in
    Nx.sum (Nx.where valid term (Nx.zeros_like term))
  in
  let nll_model = nll rates_floor in
  let nll_null = nll null_rates in
  let bps = Nx.Infix.((nll_null - nll_model) / (n_sp *$ Numeric.log2)) in
  Nx.item [] bps
