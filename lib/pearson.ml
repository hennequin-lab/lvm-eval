let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let score ~fn ~x_arg ~y_arg ~(x : Nx.host Nx.float64_t) ~(y : Nx.host Nx.float64_t) =
  Check.require_host fn x_arg x;
  Check.require_host fn y_arg y;
  Check.require_rank1 fn x_arg x;
  Check.require_rank1 fn y_arg y;
  Check.require_same_shape fn x_arg y_arg x y;
  let n = (Nx.shape x).(0) in
  if n < 2 then invalid fn (Printf.sprintf "%s needs at least two samples" x_arg);
  if not (Nx.item [] (Nx.all (Nx.isfinite x)))
  then invalid fn (x_arg ^ " contains NaN or infinity");
  if not (Nx.item [] (Nx.all (Nx.isfinite y)))
  then invalid fn (y_arg ^ " contains NaN or infinity");
  let xm = Nx.mean ~axes:[ 0 ] x
  and ym = Nx.mean ~axes:[ 0 ] y in
  let xc = Nx.Infix.(x - xm)
  and yc = Nx.Infix.(y - ym) in
  let numerator = Nx.sum Nx.Infix.(xc * yc) in
  let denominator = Nx.sqrt Nx.Infix.(Nx.sum (Nx.square xc) * Nx.sum (Nx.square yc)) in
  if Nx.item [] (Nx.equal_s denominator 0.)
  then invalid fn "correlation is undefined when an input is constant";
  Nx.item [] Nx.Infix.(numerator / denominator)
