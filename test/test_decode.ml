open Nx

let rec find_root dir =
  if Sys.file_exists (Filename.concat dir "test/fixtures/decode/expected.txt")
  then dir
  else (
    let parent = Filename.dirname dir in
    if String.equal parent dir
    then failwith "test_decode: cannot find test/fixtures/decode/expected.txt"
    else find_root parent)

let root = find_root (Sys.getcwd ())
let fixture_dir = Filename.concat root "test/fixtures/decode"
let input name = Filename.concat fixture_dir ("inputs/" ^ name ^ ".npy")
let load_float name = Nx_io.load_npy (input name) |> Nx.unpack Nx.float64
let load_bool name = Nx_io.load_npy (input name) |> Nx.unpack Nx.bool

let read_lines path =
  let channel = open_in path in
  let rec loop acc =
    match input_line channel with
    | line -> loop (line :: acc)
    | exception End_of_file ->
      close_in channel;
      List.rev acc
  in
  loop []

let words line =
  String.split_on_char ' ' line |> List.filter (fun word -> not (String.equal word ""))

let expected_cases =
  read_lines (Filename.concat fixture_dir "expected.txt")
  |> List.map (fun line ->
    match words line with
    | [ name; expected ] -> name, expected
    | _ -> failwith ("test_decode: malformed expected line: " ^ line))

let contains haystack needle =
  let length = String.length needle in
  let rec loop index =
    if index + length > String.length haystack
    then false
    else if String.equal (String.sub haystack index length) needle
    then true
    else loop (index + 1)
  in
  Int.equal length 0 || loop 0

let expect_invalid name f =
  match f () with
  | exception Invalid_argument _ -> ()
  | _ -> failwith (name ^ ": expected Invalid_argument")

let expect_invalid_contains name fragments f =
  match f () with
  | exception Invalid_argument message ->
    List.iter
      (fun fragment ->
         if not (contains message fragment)
         then
           failwith
             (Printf.sprintf
                "%s: error message %S does not name %S"
                name
                message
                fragment))
      fragments
  | _ -> failwith (name ^ ": expected Invalid_argument")

let relative_error expected got =
  if Float.equal expected 0.
  then Float.abs got
  else Float.abs (got -. expected) /. Float.abs expected

let max_relative = ref 0.
let max_absolute_zero = ref 0.

let check_value name expected got =
  let error = relative_error expected got in
  if Float.equal expected 0.
  then max_absolute_zero := Float.max !max_absolute_zero error
  else max_relative := Float.max !max_relative error;
  Printf.printf
    "%s: expected=%.17g got=%.17g rel=%.3e abs=%.3e\n%!"
    name
    expected
    got
    error
    (Float.abs (got -. expected));
  if Float.equal expected 0.
  then (if error > 1e-12 then failwith (name ^ ": absolute error exceeds 1e-12"))
  else if error > 1e-6
  then failwith (name ^ ": relative error exceeds 1e-6")

let call
      ~train_rates
      ~train_behavior
      ~eval_rates
      ~eval_behavior
      ?train_mask
      ?eval_mask
      ?alphas
      ?folds
      ()
  =
  Lvm_eval.Decode.velocity_decoding
    ~rates_train:train_rates
    ~behaviour_train:train_behavior
    ~rates_eval:eval_rates
    ~behaviour_eval:eval_behavior
    ?train_decode_mask:train_mask
    ?eval_decode_mask:eval_mask
    ?alphas
    ?folds
    ()

let () =
  List.iter
    (fun (name, expected_text) ->
       let train_rates = load_float (name ^ "_train_rates") in
       let train_behavior = load_float (name ^ "_train_behavior") in
       let eval_rates = load_float (name ^ "_eval_rates") in
       let eval_behavior = load_float (name ^ "_eval_behavior") in
       let train_mask = load_bool (name ^ "_train_mask") in
       let eval_mask = load_bool (name ^ "_eval_mask") in
       if String.equal expected_text "error"
       then
         if String.equal name "empty_mask"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.Decode.velocity_decoding"; "selects no trials" ]
             (fun () ->
                call
                  ~train_rates
                  ~train_behavior
                  ~eval_rates
                  ~eval_behavior
                  ~train_mask
                  ~eval_mask
                  ())
         else if String.equal name "few_samples"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.Ridge.select_alpha"; "two per fold" ]
             (fun () ->
                call
                  ~train_rates
                  ~train_behavior
                  ~eval_rates
                  ~eval_behavior
                  ~train_mask
                  ~eval_mask
                  ())
         else if String.equal name "all_nan_eval"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.Decode.velocity_decoding"
             ; "eval_decode_mask/behaviour_eval"
             ; "no eval samples"
             ]
             (fun () ->
                call
                  ~train_rates
                  ~train_behavior
                  ~eval_rates
                  ~eval_behavior
                  ~train_mask
                  ~eval_mask
                  ())
         else if String.equal name "one_eval_sample"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.R2.score"; "at least two rows" ]
             (fun () ->
                call
                  ~train_rates
                  ~train_behavior
                  ~eval_rates
                  ~eval_behavior
                  ~train_mask
                  ~eval_mask
                  ())
         else
           expect_invalid name (fun () ->
             call
               ~train_rates
               ~train_behavior
               ~eval_rates
               ~eval_behavior
               ~train_mask
               ~eval_mask
               ())
       else (
         let expected = float_of_string expected_text in
         let got =
           call
             ~train_rates
             ~train_behavior
             ~eval_rates
             ~eval_behavior
             ~train_mask
             ~eval_mask
             ()
         in
         check_value name expected got))
    expected_cases

let () =
  let train_rates = load_float "normal_train_rates" in
  let train_behavior = load_float "normal_train_behavior" in
  let eval_rates = load_float "normal_eval_rates" in
  let eval_behavior = load_float "normal_eval_behavior" in
  let train_mask = load_bool "normal_train_mask" in
  let eval_mask = load_bool "normal_eval_mask" in
  let rates_before = Nx.to_array train_rates in
  let train_behavior_before = Nx.to_array train_behavior in
  let eval_rates_before = Nx.to_array eval_rates in
  let eval_behavior_before = Nx.to_array eval_behavior in
  let train_mask_before = Nx.to_array train_mask in
  let eval_mask_before = Nx.to_array eval_mask in
  ignore
    (call
       ~train_rates
       ~train_behavior
       ~eval_rates
       ~eval_behavior
       ~train_mask
       ~eval_mask
       ());
  if not (Array.equal Float.equal rates_before (Nx.to_array train_rates))
  then failwith "velocity_decoding mutated rates_train";
  if not (Array.equal Float.equal train_behavior_before (Nx.to_array train_behavior))
  then failwith "velocity_decoding mutated behaviour_train";
  if not (Array.equal Float.equal eval_rates_before (Nx.to_array eval_rates))
  then failwith "velocity_decoding mutated rates_eval";
  if not (Array.equal Float.equal eval_behavior_before (Nx.to_array eval_behavior))
  then failwith "velocity_decoding mutated behaviour_eval";
  if not (Array.equal Bool.equal train_mask_before (Nx.to_array train_mask))
  then failwith "velocity_decoding mutated train_decode_mask";
  if not (Array.equal Bool.equal eval_mask_before (Nx.to_array eval_mask))
  then failwith "velocity_decoding mutated eval_decode_mask"

let () =
  let train_rates = load_float "normal_train_rates" in
  let train_behavior = load_float "normal_train_behavior" in
  let eval_rates = load_float "normal_eval_rates" in
  let eval_behavior = load_float "normal_eval_behavior" in
  let train_mask = load_bool "normal_train_mask" in
  let eval_mask = load_bool "normal_eval_mask" in
  let expected = 0.99999999998978073 in
  let got = call ~train_rates ~train_behavior ~eval_rates ~eval_behavior () in
  check_value "default mask" expected got;
  let rates_2d = create float64 [| 8; 5 |] (Array.make 40 1.) in
  let behavior_2d = create float64 [| 8; 5 |] (Array.make 40 1.) in
  expect_invalid_contains
    "rates train rank"
    [ "Lvm_eval.Decode.velocity_decoding"; "rates_train"; "[trial;time;neuron]" ]
    (fun () ->
       call
         ~train_rates:rates_2d
         ~train_behavior
         ~eval_rates
         ~eval_behavior
         ~train_mask
         ~eval_mask
         ());
  expect_invalid_contains
    "behaviour train rank"
    [ "Lvm_eval.Decode.velocity_decoding"; "behaviour_train"; "[trial;time;dim]" ]
    (fun () ->
       call
         ~train_rates
         ~train_behavior:behavior_2d
         ~eval_rates
         ~eval_behavior
         ~train_mask
         ~eval_mask
         ());
  let mask_3d = create bool [| 8; 1; 1 |] (Array.make 8 true) in
  expect_invalid_contains
    "mask rank"
    [ "Lvm_eval.Decode.velocity_decoding"; "train_decode_mask"; "[trial;mask]" ]
    (fun () ->
       call
         ~train_rates
         ~train_behavior
         ~eval_rates
         ~eval_behavior
         ~train_mask:mask_3d
         ~eval_mask
         ());
  expect_invalid_contains
    "only one mask"
    [ "Lvm_eval.Decode.velocity_decoding"; "both be present" ]
    (fun () ->
       call ~train_rates ~train_behavior ~eval_rates ~eval_behavior ~train_mask ());
  let eval_rates_wrong = create float64 [| 4; 5; 3 |] (Array.make 60 1.) in
  expect_invalid_contains
    "eval neuron count"
    [ "Lvm_eval.Decode.velocity_decoding"; "neuron count" ]
    (fun () ->
       call
         ~train_rates
         ~train_behavior
         ~eval_rates:eval_rates_wrong
         ~eval_behavior
         ~train_mask
         ~eval_mask
         ());
  expect_invalid_contains "folds" [ "Lvm_eval.Ridge.select_alpha"; "folds" ] (fun () ->
    call
      ~train_rates
      ~train_behavior
      ~eval_rates
      ~eval_behavior
      ~train_mask
      ~eval_mask
      ~folds:1
      ());
  expect_invalid_contains
    "empty alpha grid"
    [ "Lvm_eval.Ridge.select_alpha"; "alpha grid is empty" ]
    (fun () ->
       call
         ~train_rates
         ~train_behavior
         ~eval_rates
         ~eval_behavior
         ~train_mask
         ~eval_mask
         ~alphas:[||]
         ())

let () =
  Printf.printf
    "decode: max relative error=%.3e max absolute error (expected zero)=%.3e\n%!"
    !max_relative
    !max_absolute_zero
