let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let bps ~fn ~rates_arg ~spikes_arg ~mask_arg ~rates ~spikes ?mask () =
  Check.require_host fn rates_arg rates;
  Check.require_host fn spikes_arg spikes;
  Option.iter (Check.require_host fn mask_arg) mask;
  Check.require_rank3 fn rates_arg rates;
  Check.require_rank3 fn spikes_arg spikes;
  Check.require_same_shape fn rates_arg spikes_arg rates spikes;
  Option.iter
    (fun mask -> Check.require_same_shape fn mask_arg spikes_arg mask spikes)
    mask;
  let valid = Mask.valid ~fn ?mask spikes in
  let counts = Mask.count_per_neuron valid in
  let spikes_valid = Mask.selected valid spikes in
  let rates_valid = Mask.selected valid rates in
  let all_finite_spikes = Nx.item [] (Nx.all (Nx.isfinite spikes_valid)) in
  if not all_finite_spikes
  then invalid fn (spikes_arg ^ " contains NaN or infinity on a scored entry");
  let all_finite_rates = Nx.item [] (Nx.all (Nx.isfinite rates_valid)) in
  if not all_finite_rates
  then invalid fn (rates_arg ^ " contains NaN or infinity on a scored entry");
  if Nx.item [] (Nx.any (Nx.less_s spikes_valid 0.))
  then invalid fn (spikes_arg ^ " must be nonnegative");
  if Nx.item [] (Nx.any (Nx.less_s rates_valid 0.))
  then invalid fn (rates_arg ^ " must be nonnegative");
  if not (Nx.item [] (Nx.all (Nx.equal spikes_valid (Nx.floor spikes_valid))))
  then invalid fn (spikes_arg ^ " must be integer-valued");
  let n_sp = Mask.total valid spikes in
  if Nx.item [] (Nx.less_equal_s n_sp 0.)
  then invalid fn "no scored spikes (zero total spikes)";
  let rates_floor = Numeric.floor_zero_rates rates_valid in
  let safe_counts =
    Nx.where (Nx.greater counts (Nx.scalar Nx.float64 0.)) counts (Nx.ones_like counts)
  in
  let null_rates = Nx.Infix.(Mask.sum_per_neuron valid spikes / safe_counts) in
  let null_rates = Numeric.floor_zero_rates null_rates in
  let n = (Nx.shape spikes).(2) in
  let null_rates =
    null_rates |> Nx.reshape [| 1; 1; n |] |> Nx.broadcast_to (Nx.shape spikes)
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

let co_smoothing ~rates_heldout ~spikes_heldout ?mask_heldout () =
  bps
    ~fn:"Lvm_eval.Bits.co_smoothing"
    ~rates_arg:"rates_heldout"
    ~spikes_arg:"spikes_heldout"
    ~mask_arg:"mask_heldout"
    ~rates:rates_heldout
    ~spikes:spikes_heldout
    ?mask:mask_heldout
    ()

let forward_prediction ~rates_forward ~spikes_forward ?mask_forward () =
  bps
    ~fn:"Lvm_eval.Bits.forward_prediction"
    ~rates_arg:"rates_forward"
    ~spikes_arg:"spikes_forward"
    ~mask_arg:"mask_forward"
    ~rates:rates_forward
    ~spikes:spikes_forward
    ?mask:mask_forward
    ()
