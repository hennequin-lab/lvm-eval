let score ~(y_true : Nx.host Nx.float64_t) ~(y_pred : Nx.host Nx.float64_t) =
  let fn = "Lvm_eval.R2.score" in
  Check.require_host fn "y_true" y_true;
  Check.require_host fn "y_pred" y_pred;
  Check.require_rank2 fn "y_true" y_true;
  Check.require_rank2 fn "y_pred" y_pred;
  Check.require_same_shape fn "y_true" "y_pred" y_true y_pred;
  if (Nx.shape y_true).(0) < 2
  then invalid_arg (fn ^ ": y_true must have at least two rows");
  if Int.equal (Nx.shape y_true).(1) 0
  then invalid_arg (fn ^ ": y_true must have at least one neuron");
  let mean_true = Nx.mean ~axes:[ 0 ] y_true in
  let sse = Nx.sum ~axes:[ 0 ] (Nx.square Nx.Infix.(y_true - y_pred)) in
  let sst = Nx.sum ~axes:[ 0 ] (Nx.square Nx.Infix.(y_true - mean_true)) in
  let ratio =
    Nx.where
      (Nx.greater_s sst 0.)
      Nx.Infix.(sse / sst)
      (Nx.where (Nx.greater_s sse 0.) (Nx.ones_like sse) (Nx.zeros_like sse))
  in
  let r2 = Nx.Infix.(Nx.ones_like ratio - ratio) in
  Nx.item [] (Nx.mean r2)
