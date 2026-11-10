let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let shape3 fn arg x =
  match Nx.shape x with
  | [| a; b; c |] -> a, b, c
  | _ -> assert false

let flatten_3d fn arg x =
  let n = (Nx.shape x).(2) in
  Nx.reshape [| -1; n |] x

let select_trials fn arg indices x =
  if List.is_empty indices then invalid fn (arg ^ " selects no trials");
  Nx.slice [ L indices ] x

let drop_first_column_nan fn arg behaviour =
  let first = Nx.slice [ A; I 0 ] behaviour in
  let valid = Nx.logical_not (Nx.isnan first) in
  let selected = Nx.slice [ M valid ] behaviour in
  if not (Nx.item [] (Nx.all (Nx.isfinite selected)))
  then invalid fn (arg ^ " contains NaN or infinity after row dropping");
  selected

let all_true_trials trials = Nx.create Nx.bool [| trials; 1 |] (Array.make trials true)

let velocity_decoding
      ~rates_train
      ~behaviour_train
      ~rates_eval
      ~behaviour_eval
      ?train_decode_mask
      ?eval_decode_mask
      ?(alphas = Ridge.default_alphas)
      ?(folds = 5)
      ()
  =
  let fn = "Lvm_eval.Decode.velocity_decoding" in
  Check.require_host fn "rates_train" rates_train;
  Check.require_host fn "behaviour_train" behaviour_train;
  Check.require_host fn "rates_eval" rates_eval;
  Check.require_host fn "behaviour_eval" behaviour_eval;
  Check.require_rank3 fn "rates_train" rates_train;
  Check.require_rank3 ~axes:"[trial;time;dim]" fn "behaviour_train" behaviour_train;
  Check.require_rank3 fn "rates_eval" rates_eval;
  Check.require_rank3 ~axes:"[trial;time;dim]" fn "behaviour_eval" behaviour_eval;
  let btr, ttr, ntr = shape3 fn "rates_train" rates_train in
  let bte, tte, net = shape3 fn "rates_eval" rates_eval in
  let bbtr, tbtr, dtr = shape3 fn "behaviour_train" behaviour_train in
  let bbte, tbte, dte = shape3 fn "behaviour_eval" behaviour_eval in
  if (not (Int.equal btr bbtr)) || not (Int.equal ttr tbtr)
  then invalid fn "rates_train and behaviour_train must share trial/time";
  if (not (Int.equal bte bbte)) || not (Int.equal tte tbte)
  then invalid fn "rates_eval and behaviour_eval must share trial/time";
  if not (Int.equal ntr net)
  then invalid fn "rates_train and rates_eval must have the same neuron count";
  if not (Int.equal dtr dte)
  then invalid fn "behaviour_train and behaviour_eval must have the same dim";
  if Int.equal ntr 0 || Int.equal dtr 0
  then invalid fn "rates and behaviour must have non-zero feature dimensions";
  let train_mask, eval_mask =
    match train_decode_mask, eval_decode_mask with
    | None, None -> all_true_trials btr, all_true_trials bte
    | Some train_mask, Some eval_mask ->
      Check.require_host fn "train_decode_mask" train_mask;
      Check.require_host fn "eval_decode_mask" eval_mask;
      Check.require_rank2 ~axes:"[trial;mask]" fn "train_decode_mask" train_mask;
      Check.require_rank2 ~axes:"[trial;mask]" fn "eval_decode_mask" eval_mask;
      (match Nx.shape train_mask with
       | [| b; k |] when Int.equal b btr ->
         (match Nx.shape eval_mask with
          | [| b'; k' |] when Int.equal b' bte && Int.equal k' k -> ()
          | shape ->
            invalid
              fn
              (Printf.sprintf
                 "eval_decode_mask must have shape [%d;%d], got %s"
                 bte
                 k
                 (Format.asprintf "%a" Nx.pp_shape shape)))
       | shape ->
         invalid
           fn
           (Printf.sprintf
              "train_decode_mask must have shape [%d;K], got %s"
              btr
              (Format.asprintf "%a" Nx.pp_shape shape)));
      train_mask, eval_mask
    | Some _, None | None, Some _ ->
      invalid
        fn
        "train_decode_mask and eval_decode_mask must both be present or both be absent"
  in
  let k = (Nx.shape train_mask).(1) in
  if Int.equal k 0 then invalid fn "decode masks must have at least one column";
  let train_flags = Nx.to_array train_mask
  and eval_flags = Nx.to_array eval_mask in
  let indices flags trials column =
    List.init trials (fun trial ->
      let position = (trial * k) + column in
      if flags.(position) then Some trial else None)
    |> List.filter_map (fun trial -> trial)
  in
  let scores =
    List.init k (fun column ->
      let train_indices = indices train_flags btr column
      and eval_indices = indices eval_flags bte column in
      let train_x =
        select_trials fn "train_decode_mask" train_indices rates_train
        |> flatten_3d fn "rates_train"
      and train_y =
        select_trials fn "train_decode_mask" train_indices behaviour_train
        |> flatten_3d fn "behaviour_train"
      and eval_x =
        select_trials fn "eval_decode_mask" eval_indices rates_eval
        |> flatten_3d fn "rates_eval"
      and eval_y =
        select_trials fn "eval_decode_mask" eval_indices behaviour_eval
        |> flatten_3d fn "behaviour_eval"
      in
      let valid = Nx.logical_not (Nx.isnan (Nx.slice [ A; I 0 ] train_y)) in
      let train_x = Nx.slice [ M valid ] train_x
      and train_y = drop_first_column_nan fn "behaviour_train" train_y in
      let valid_eval = Nx.logical_not (Nx.isnan (Nx.slice [ A; I 0 ] eval_y)) in
      let eval_x = Nx.slice [ M valid_eval ] eval_x
      and eval_y = drop_first_column_nan fn "behaviour_eval" eval_y in
      if Int.equal (Nx.shape train_x).(0) 0
      then invalid fn "train_decode_mask/behaviour_train retains no training samples";
      if Int.equal (Nx.shape eval_x).(0) 0
      then invalid fn "eval_decode_mask/behaviour_eval retains no eval samples";
      if not (Nx.item [] (Nx.all (Nx.isfinite train_x)))
      then invalid fn "rates_train contains NaN or infinity after row dropping";
      if not (Nx.item [] (Nx.all (Nx.isfinite eval_x)))
      then invalid fn "rates_eval contains NaN or infinity after row dropping";
      let alpha = Ridge.select_alpha ~x:train_x ~y:train_y ~alphas ~folds in
      let model = Ridge.fit ~x:train_x ~y:train_y ~alpha in
      let predicted = Ridge.predict model eval_x in
      R2.score ~y_true:eval_y ~y_pred:predicted)
  in
  List.fold_left ( +. ) 0. scores /. Float.of_int (List.length scores)
