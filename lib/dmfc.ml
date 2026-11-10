let invalid fn message = invalid_arg (fn ^ ": " ^ message)

let speed_tp_code ~rates_eval ~mask ~conditions ~tp () =
  let fn = "Lvm_eval.Dmfc.speed_tp_code" in
  Check.require_host fn "rates_eval" rates_eval;
  Check.require_host fn "mask" mask;
  Check.require_rank3 fn "rates_eval" rates_eval;
  Check.require_rank2 ~axes:"[trial;time]" fn "mask" mask;
  let b, t =
    match Nx.shape rates_eval with
    | [| b; t; _ |] -> b, t
    | _ -> assert false
  in
  (match Nx.shape mask with
   | [| b'; t' |] when Int.equal b b' && Int.equal t t' -> ()
   | shape ->
     invalid
       fn
       (Printf.sprintf
          "mask must have shape [%d;%d], got %s"
          b
          t
          (Format.asprintf "%a" Nx.pp_shape shape)));
  if not (Int.equal (Array.length conditions) b)
  then
    invalid
      fn
      (Printf.sprintf
         "conditions must have %d entries, got %d"
         b
         (Array.length conditions));
  if not (Int.equal (Array.length tp) b)
  then invalid fn (Printf.sprintf "tp must have %d entries, got %d" b (Array.length tp));
  let counts = Nx.sum ~axes:[ 1 ] (Nx.cast Nx.float64 mask) |> Nx.to_array in
  Array.iteri
    (fun trial count ->
       if count <= 0.
       then
         invalid
           fn
           (Printf.sprintf "mask must select at least one time point for trial %d" trial))
    counts;
  let speeds =
    Array.init b (fun trial ->
      let selected =
        Nx.slice [ M (Nx.slice [ I trial ] mask) ] (Nx.slice [ I trial ] rates_eval)
      in
      let n_t = (Nx.shape selected).(0) in
      let last = n_t - 1 in
      let diff =
        Nx.Infix.(
          Nx.slice [ R (1, n_t); A ] selected - Nx.slice [ R (0, last); A ] selected)
      in
      let norms = Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.square diff)) in
      Nx.item [] (Nx.mean norms))
  in
  let condition_list = Array.to_list conditions |> List.sort_uniq Int.compare in
  if List.is_empty condition_list
  then invalid fn "conditions must contain at least one trial";
  let correlations =
    List.map
      (fun condition ->
         let indices =
           List.init b (fun trial -> trial)
           |> List.filter (fun trial -> Int.equal conditions.(trial) condition)
         in
         let x =
           Nx.create
             Nx.float64
             [| List.length indices |]
             (Array.of_list (List.map (fun trial -> speeds.(trial)) indices))
         and y =
           Nx.create
             Nx.float64
             [| List.length indices |]
             (Array.of_list (List.map (fun trial -> tp.(trial)) indices))
         in
         Pearson.score ~fn ~x_arg:"speed" ~y_arg:"tp" ~x ~y)
      condition_list
  in
  List.fold_left ( +. ) 0. correlations /. Float.of_int (List.length correlations)

let speed_tp_global ~rates_eval ~mask ~tp () =
  let fn = "Lvm_eval.Dmfc.speed_tp_global" in
  Check.require_host fn "rates_eval" rates_eval;
  Check.require_host fn "mask" mask;
  Check.require_rank3 fn "rates_eval" rates_eval;
  Check.require_rank2 ~axes:"[trial;time]" fn "mask" mask;
  let b, t =
    match Nx.shape rates_eval with
    | [| b; t; _ |] -> b, t
    | _ -> assert false
  in
  (match Nx.shape mask with
   | [| b'; t' |] when Int.equal b b' && Int.equal t t' -> ()
   | shape ->
     invalid
       fn
       (Printf.sprintf
          "mask must have shape [%d;%d], got %s"
          b
          t
          (Format.asprintf "%a" Nx.pp_shape shape)));
  if not (Int.equal (Array.length tp) b)
  then invalid fn (Printf.sprintf "tp must have %d entries, got %d" b (Array.length tp));
  let counts = Nx.sum ~axes:[ 1 ] (Nx.cast Nx.float64 mask) |> Nx.to_array in
  Array.iteri
    (fun trial count ->
       if count <= 0.
       then
         invalid
           fn
           (Printf.sprintf "mask must select at least one time point for trial %d" trial))
    counts;
  let speeds =
    Array.init b (fun trial ->
      let selected =
        Nx.slice [ M (Nx.slice [ I trial ] mask) ] (Nx.slice [ I trial ] rates_eval)
      in
      let n_t = (Nx.shape selected).(0) in
      let last = n_t - 1 in
      let diff =
        Nx.Infix.(
          Nx.slice [ R (1, n_t); A ] selected - Nx.slice [ R (0, last); A ] selected)
      in
      let norms = Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.square diff)) in
      Nx.item [] (Nx.mean norms))
  in
  let x = Nx.create Nx.float64 [| b |] speeds in
  let y = Nx.create Nx.float64 [| b |] (Array.copy tp) in
  Pearson.score ~fn ~x_arg:"speed" ~y_arg:"tp" ~x ~y
