let invalid fn message = invalid_arg (fn ^ ": " ^ message)

type t =
  { x_mean : Nx.float64_t
  ; y_mean : Nx.float64_t
  ; coef : Nx.float64_t
  }

let default_alphas =
  Array.init 9 (fun index -> Float.pow 10. (-4. +. (0.5 *. Float.of_int index)))

let require_matrix fn arg x =
  Check.require_host fn arg x;
  Check.require_rank2 ~axes:"[sample;feature]" fn arg x

let require_finite fn arg x =
  if not (Nx.item [] (Nx.all (Nx.isfinite x)))
  then invalid fn (arg ^ " contains NaN or infinity")

let svd_solve ~x ~y ~alpha =
  let u, singular, vt = Nx.svd x in
  let uty = Nx.matmul (Nx.transpose u) y in
  let scale =
    Nx.where
      (Nx.greater_s singular 1e-15)
      Nx.Infix.(singular / ((singular * singular) +$ alpha))
      (Nx.zeros_like singular)
  in
  let rank = (Nx.shape singular).(0) in
  Nx.matmul (Nx.transpose vt) Nx.Infix.(Nx.reshape [| rank; 1 |] scale * uty)

let fit ~x ~y ~alpha =
  let fn = "Lvm_eval.Ridge.fit" in
  require_matrix fn "x" x;
  require_matrix fn "y" y;
  require_finite fn "x" x;
  require_finite fn "y" y;
  if (not (Float.is_finite alpha)) || alpha <= 0.
  then invalid fn (Printf.sprintf "alpha must be finite and positive, got %g" alpha);
  if not (Int.equal (Nx.shape x).(0) (Nx.shape y).(0))
  then
    invalid
      fn
      (Printf.sprintf
         "x and y have different sample counts, %d and %d"
         (Nx.shape x).(0)
         (Nx.shape y).(0));
  let d = (Nx.shape x).(1) in
  if Int.equal d 0 then invalid fn "x must have at least one feature";
  let x_mean = Nx.mean ~axes:[ 0 ] x
  and y_mean = Nx.mean ~axes:[ 0 ] y in
  let xc = Nx.Infix.(x - x_mean)
  and yc = Nx.Infix.(y - y_mean) in
  let gram = Nx.Infix.(Nx.matmul (Nx.transpose xc) xc + (Nx.eye Nx.float64 d *$ alpha)) in
  let rhs = Nx.matmul (Nx.transpose xc) yc in
  let factor = Nx.cholesky gram in
  let coef =
    if Nx.item [] (Nx.all (Nx.isfinite factor))
    then (
      let forward = Nx.solve_triangular ~upper:false factor rhs in
      Nx.solve_triangular ~upper:true (Nx.transpose factor) forward)
    else svd_solve ~x:xc ~y:yc ~alpha
  in
  { x_mean; y_mean; coef }

let predict model x =
  let fn = "Lvm_eval.Ridge.predict" in
  require_matrix fn "x" x;
  require_finite fn "x" x;
  if not (Int.equal (Nx.shape x).(1) (Nx.shape model.coef).(0))
  then
    invalid
      fn
      (Printf.sprintf
         "x has %d features, expected %d"
         (Nx.shape x).(1)
         (Nx.shape model.coef).(0));
  Nx.Infix.(Nx.matmul (x - model.x_mean) model.coef + model.y_mean)

let fold_range ~n ~folds fold =
  let size = (n / folds) + if fold < n mod folds then 1 else 0 in
  let start = (fold * (n / folds)) + Int.min fold (n mod folds) in
  start, start + size

let select_alpha ~x ~y ~alphas ~folds =
  let fn = "Lvm_eval.Ridge.select_alpha" in
  require_matrix fn "x" x;
  require_matrix fn "y" y;
  require_finite fn "x" x;
  require_finite fn "y" y;
  if Int.equal (Array.length alphas) 0 then invalid fn "the alpha grid is empty";
  if folds < 2 then invalid fn "folds must be at least 2";
  let n = (Nx.shape x).(0) in
  if n < 2 * folds
  then
    invalid
      fn
      (Printf.sprintf "training samples (%d) fewer than two per fold (%d folds)" n folds);
  let score alpha =
    let total = ref 0. in
    for fold = 0 to folds - 1 do
      let start, stop = fold_range ~n ~folds fold in
      let validation = List.init (stop - start) (fun i -> start + i) in
      let training =
        List.init n (fun i -> i) |> List.filter (fun i -> i < start || i >= stop)
      in
      let model =
        fit ~x:(Nx.slice [ L training ] x) ~y:(Nx.slice [ L training ] y) ~alpha
      in
      let predicted = predict model (Nx.slice [ L validation ] x) in
      total := !total +. R2.score ~y_true:(Nx.slice [ L validation ] y) ~y_pred:predicted
    done;
    !total /. Float.of_int folds
  in
  let best = ref None in
  Array.iter
    (fun alpha ->
       let candidate = score alpha in
       match !best with
       | Some (_, best_score) when Float.compare candidate best_score <= 0 -> ()
       | _ -> best := Some (alpha, candidate))
    alphas;
  match !best with
  | Some (alpha, _) -> alpha
  | None -> invalid fn "the alpha grid is empty"
