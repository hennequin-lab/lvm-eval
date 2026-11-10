let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let shift_trial ~fn ~jitter ~trial =
  let t = (Nx.shape trial).(0) in
  let n = (Nx.shape trial).(1) in
  if jitter <= -t || jitter >= t
  then
    invalid fn (Printf.sprintf "jitter %d is outside (-time, time) for time=%d" jitter t);
  if Int.equal jitter 0
  then trial
  else if jitter > 0
  then (
    let pad = Nx.full Nx.float64 [| jitter; n |] Float.nan in
    let body = Nx.slice [ R (0, t - jitter); A ] trial in
    Nx.concatenate ~axis:0 [ pad; body ])
  else (
    let shift = -jitter in
    let body = Nx.slice [ R (shift, t); A ] trial in
    let pad = Nx.full Nx.float64 [| shift; n |] Float.nan in
    Nx.concatenate ~axis:0 [ body; pad ])

let r2 ~rates_eval ~psth ~eval_cond_idx ?jitter () =
  let fn = "Lvm_eval.Psth.r2" in
  Check.require_host fn "rates_eval" rates_eval;
  Check.require_host fn "psth" psth;
  Check.require_rank3 fn "rates_eval" rates_eval;
  Check.require_rank3 fn "psth" psth;
  let b, t, n =
    match Nx.shape rates_eval with
    | [| b; t; n |] -> b, t, n
    | _ -> assert false
  and c, tp, np =
    match Nx.shape psth with
    | [| c; tp; np |] -> c, tp, np
    | _ -> assert false
  in
  if not (Int.equal t tp)
  then
    invalid
      fn
      (Printf.sprintf
         "rates_eval and psth must have the same time length, got %d and %d"
         t
         tp);
  if not (Int.equal n np)
  then
    invalid
      fn
      (Printf.sprintf
         "rates_eval and psth must have the same neuron count, got %d and %d"
         n
         np);
  if not (Int.equal (Array.length eval_cond_idx) c)
  then
    invalid
      fn
      (Printf.sprintf
         "eval_cond_idx must have %d entries, got %d"
         c
         (Array.length eval_cond_idx));
  let jitter =
    match jitter with
    | None -> Array.make b 0
    | Some jitter ->
      if not (Int.equal (Array.length jitter) b)
      then
        invalid
          fn
          (Printf.sprintf "jitter must have %d entries, got %d" b (Array.length jitter));
      jitter
  in
  Array.iter
    (fun indices ->
       Array.iter
         (fun index ->
            if index < 0 || index >= b
            then
              invalid
                fn
                (Printf.sprintf
                   "eval_cond_idx contains trial index %d outside [0,%d)"
                   index
                   b))
         indices)
    eval_cond_idx;
  let true_rows = ref []
  and pred_rows = ref [] in
  for condition = 0 to c - 1 do
    let indices = eval_cond_idx.(condition) in
    if Array.length indices > 0
    then (
      let psth_c = Nx.slice [ I condition ] psth in
      let psth_first = Nx.slice [ I condition; A; I 0 ] psth in
      let valid_rows = Nx.logical_not (Nx.isnan psth_first) in
      let pred_c =
        Array.to_list indices
        |> List.map (fun index ->
          let trial = Nx.slice [ I index ] rates_eval in
          shift_trial ~fn ~jitter:jitter.(index) ~trial)
        |> Nx.stack ~axis:0
        |> Nx.mean ~axes:[ 0 ]
      in
      let true_c = Nx.slice [ M valid_rows ] psth_c
      and pred_c = Nx.slice [ M valid_rows ] pred_c in
      if not (Nx.item [] (Nx.all (Nx.isfinite true_c)))
      then invalid fn "psth contains NaN or infinity in a selected row";
      if not (Nx.item [] (Nx.all (Nx.isfinite pred_c)))
      then invalid fn "rates_eval contains NaN or infinity in a selected row";
      if (Nx.shape true_c).(0) > 0
      then (
        true_rows := true_c :: !true_rows;
        pred_rows := pred_c :: !pred_rows))
  done;
  match List.rev !true_rows, List.rev !pred_rows with
  | [], [] -> invalid fn "no valid PSTH rows"
  | true_rows, pred_rows ->
    let y_true = Nx.concatenate ~axis:0 true_rows
    and y_pred = Nx.concatenate ~axis:0 pred_rows in
    R2.score ~y_true ~y_pred
