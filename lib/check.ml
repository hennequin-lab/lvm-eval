let shape_to_string shape = Format.asprintf "%a" Nx.pp_shape shape

let require_rank2 fn arg x =
  match Nx.shape x with
  | [| _; _ |] -> ()
  | shape ->
    invalid_arg
      (Printf.sprintf
         "%s: %s must have shape [row;neuron], got %s"
         fn
         arg
         (shape_to_string shape))

let require_rank3 ?(axes = "[trial;time;neuron]") fn arg x =
  match Nx.shape x with
  | [| _; _; _ |] -> ()
  | shape ->
    invalid_arg
      (Printf.sprintf
         "%s: %s must have shape %s, got %s"
         fn
         arg
         axes
         (shape_to_string shape))

let require_same_shape fn arg1 arg2 x y =
  let sx = Nx.shape x
  and sy = Nx.shape y in
  if not (Array.equal Int.equal sx sy)
  then
    invalid_arg
      (Printf.sprintf
         "%s: %s and %s must have the same shape, got %s and %s"
         fn
         arg1
         arg2
         (shape_to_string sx)
         (shape_to_string sy))

let require_host fn arg x =
  let placement = Nx.placement x in
  if not (Nx.Placement.equal placement Nx.Placement.host)
  then
    invalid_arg
      (Format.asprintf
         "%s: %s must be on Placement.host, got %a"
         fn
         arg
         Nx.Placement.pp
         placement)
