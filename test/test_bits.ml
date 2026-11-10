open Nx

let rec find_root dir =
  if Sys.file_exists (Filename.concat dir "test/fixtures/bits/expected.txt")
  then dir
  else (
    let parent = Filename.dirname dir in
    if String.equal parent dir
    then failwith "test_bits: cannot find test/fixtures/bits/expected.txt"
    else find_root parent)

let root = find_root (Sys.getcwd ())
let fixture_dir = Filename.concat root "test/fixtures/bits"
let forward_dir = Filename.concat root "test/fixtures/forward"
let input name = Filename.concat fixture_dir ("inputs/" ^ name ^ ".npy")
let forward_input name = Filename.concat forward_dir ("inputs/" ^ name ^ ".npy")
let load_float name = Nx_io.load_npy (input name) |> Nx.unpack Nx.float64
let load_bool name = Nx_io.load_npy (input name) |> Nx.unpack Nx.bool
let load_forward_float name = Nx_io.load_npy (forward_input name) |> Nx.unpack Nx.float64
let load_forward_bool name = Nx_io.load_npy (forward_input name) |> Nx.unpack Nx.bool

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

let expected_cases path =
  read_lines path
  |> List.map (fun line ->
    match words line with
    | [ name; expected ] -> name, expected
    | _ -> failwith ("test_bits: malformed expected line: " ^ line))

let bits_cases = expected_cases (Filename.concat fixture_dir "expected.txt")
let forward_cases = expected_cases (Filename.concat forward_dir "expected.txt")

let call ~rates ~spikes ?mask () =
  Lvm_eval.Bits.co_smoothing
    ~rates_heldout:rates
    ~spikes_heldout:spikes
    ?mask_heldout:mask
    ()

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

let check_unchanged_float name before after =
  if not (Array.equal Float.equal before after)
  then failwith (name ^ " mutated a float64 input")

let check_unchanged_bool name before after =
  if not (Array.equal Bool.equal before after)
  then failwith (name ^ " mutated a bool input")

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

let () =
  List.iter
    (fun (name, expected_text) ->
       let rates = load_float (name ^ "_rates") in
       let spikes = load_float (name ^ "_spikes") in
       let mask =
         if Sys.file_exists (input (name ^ "_mask"))
         then Some (load_bool (name ^ "_mask"))
         else None
       in
       if String.equal expected_text "error"
       then
         if
           List.exists
             (fun error_case -> String.equal name error_case)
             [ "zero_total_spikes"; "all_false_mask" ]
         then
           expect_invalid_contains name [ "no scored spikes" ] (fun () ->
             call ~rates ~spikes ?mask ())
         else expect_invalid name (fun () -> call ~rates ~spikes ?mask ())
       else (
         let expected = float_of_string expected_text in
         let got = call ~rates ~spikes ?mask () in
         check_value name expected got))
    bits_cases

let () =
  (* The reference mutates zero rates in place; lvm-eval must not. *)
  let rates = load_float "zero_rate_rates" in
  let spikes = load_float "zero_rate_spikes" in
  let rates_before = Nx.to_array rates in
  let spikes_before = Nx.to_array spikes in
  ignore (call ~rates ~spikes ());
  check_unchanged_float "co_smoothing rates_heldout" rates_before (Nx.to_array rates);
  check_unchanged_float "co_smoothing spikes_heldout" spikes_before (Nx.to_array spikes)

let () =
  (* An explicit mask derived from NaN spikes must agree with the NaN path. *)
  let rates = load_float "nan_spikes_rates" in
  let spikes = load_float "nan_spikes_spikes" in
  let mask = load_bool "nan_spikes_mask" in
  let rates_before = Nx.to_array rates in
  let spikes_before = Nx.to_array spikes in
  let mask_before = Nx.to_array mask in
  let without = call ~rates ~spikes () in
  let with_mask = call ~rates ~spikes ~mask () in
  check_unchanged_float
    "co_smoothing masked rates_heldout"
    rates_before
    (Nx.to_array rates);
  check_unchanged_float "co_smoothing NaN spikes" spikes_before (Nx.to_array spikes);
  check_unchanged_bool "co_smoothing mask_heldout" mask_before (Nx.to_array mask);
  if relative_error without with_mask > 1e-12
  then failwith "explicit mask and NaN-spike paths disagree"

let () =
  (* Rank and shape errors name the function and argument. *)
  let rates_3d = create float64 [| 1; 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let spikes_3d = create float64 [| 1; 2; 2 |] [| 1.; 0.; 2.; 0. |] in
  let rates_2d = create float64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let spikes_2d = create float64 [| 2; 2 |] [| 1.; 0.; 2.; 0. |] in
  let spikes_wrong_neurons = create float64 [| 1; 2; 1 |] [| 1.; 0. |] in
  let mask_wrong_neurons = create bool [| 1; 2; 1 |] [| true; false |] in
  let inf_spikes = create float64 [| 1; 2; 2 |] [| 1.; 0.; Float.infinity; 0. |] in
  expect_invalid_contains
    "rank"
    [ "Lvm_eval.Bits.co_smoothing"; "rates_heldout"; "[trial;time;neuron]" ]
    (fun () -> call ~rates:rates_2d ~spikes:spikes_2d ());
  expect_invalid_contains
    "shape"
    [ "Lvm_eval.Bits.co_smoothing"; "rates_heldout"; "spikes_heldout" ]
    (fun () -> call ~rates:rates_3d ~spikes:spikes_wrong_neurons ());
  expect_invalid_contains "mask shape" [ "mask_heldout"; "spikes_heldout" ] (fun () ->
    call ~rates:rates_3d ~spikes:spikes_3d ~mask:mask_wrong_neurons ());
  expect_invalid_contains
    "inf spikes"
    [ "Lvm_eval.Bits.co_smoothing"; "spikes_heldout"; "infinity" ]
    (fun () -> call ~rates:rates_3d ~spikes:inf_spikes ())

let () =
  let call_forward ~rates ~spikes ?mask () =
    Lvm_eval.Bits.forward_prediction
      ~rates_forward:rates
      ~spikes_forward:spikes
      ?mask_forward:mask
      ()
  in
  List.iter
    (fun (name, expected_text) ->
       let rates = load_forward_float (name ^ "_rates") in
       let spikes = load_forward_float (name ^ "_spikes") in
       let mask =
         if Sys.file_exists (forward_input (name ^ "_mask"))
         then Some (load_forward_bool (name ^ "_mask"))
         else None
       in
       let expected = float_of_string expected_text in
       let got = call_forward ~rates ~spikes ?mask () in
       check_value ("forward." ^ name) expected got)
    forward_cases

let () =
  let call_forward ~rates ~spikes ?mask () =
    Lvm_eval.Bits.forward_prediction
      ~rates_forward:rates
      ~spikes_forward:spikes
      ?mask_forward:mask
      ()
  in
  let rates = load_forward_float "forward_nan_rates" in
  let spikes = load_forward_float "forward_nan_spikes" in
  let mask = load_forward_bool "forward_nan_mask" in
  let rates_before = Nx.to_array rates in
  let spikes_before = Nx.to_array spikes in
  let mask_before = Nx.to_array mask in
  let without = call_forward ~rates ~spikes () in
  let with_mask = call_forward ~rates ~spikes ~mask () in
  check_unchanged_float
    "forward_prediction rates_forward"
    rates_before
    (Nx.to_array rates);
  check_unchanged_float
    "forward_prediction spikes_forward"
    spikes_before
    (Nx.to_array spikes);
  check_unchanged_bool "forward_prediction mask_forward" mask_before (Nx.to_array mask);
  if relative_error without with_mask > 1e-12
  then failwith "forward explicit mask and NaN-spike paths disagree"

let () =
  let call_forward ~rates ~spikes ?mask () =
    Lvm_eval.Bits.forward_prediction
      ~rates_forward:rates
      ~spikes_forward:spikes
      ?mask_forward:mask
      ()
  in
  let rates = create float64 [| 1; 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let spikes = create float64 [| 1; 2; 2 |] [| 1.; 0.; 2.; 0. |] in
  let spikes_wrong = create float64 [| 1; 2; 3 |] [| 1.; 0.; 2.; 0.; 1.; 1. |] in
  let rates_2d = create float64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let spikes_2d = create float64 [| 2; 2 |] [| 1.; 0.; 2.; 0. |] in
  let mask_wrong =
    create bool [| 1; 2; 3 |] [| true; false; true; false; true; false |]
  in
  expect_invalid_contains
    "forward rank"
    [ "Lvm_eval.Bits.forward_prediction"; "rates_forward"; "[trial;time;neuron]" ]
    (fun () -> call_forward ~rates:rates_2d ~spikes:spikes_2d ());
  expect_invalid_contains
    "forward shape"
    [ "Lvm_eval.Bits.forward_prediction"; "rates_forward"; "spikes_forward" ]
    (fun () -> call_forward ~rates ~spikes:spikes_wrong ());
  expect_invalid_contains
    "forward mask shape"
    [ "mask_forward"; "spikes_forward" ]
    (fun () -> call_forward ~rates ~spikes ~mask:mask_wrong ())

let () =
  Printf.printf
    "bits/forward: max relative error=%.3e max absolute error (expected zero)=%.3e\n%!"
    !max_relative
    !max_absolute_zero
