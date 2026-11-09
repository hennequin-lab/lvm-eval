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
    | _ -> failwith ("test_bits: malformed expected line: " ^ line))

let call ~rates ~spikes ?mask () =
  Lvm_eval.Bits.co_smoothing
    ~rates_heldout:rates
    ~spikes_heldout:spikes
    ?mask_heldout:mask
    ()

let expect_invalid name f =
  match f () with
  | exception Invalid_argument _ -> ()
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
       then expect_invalid name (fun () -> call ~rates ~spikes ?mask ())
       else (
         let expected = float_of_string expected_text in
         let got = call ~rates ~spikes ?mask () in
         check_value name expected got))
    expected_cases

let () =
  (* The reference mutates zero rates in place; lvm-eval must not. *)
  let rates = load_float "zero_rate_rates" in
  let spikes = load_float "zero_rate_spikes" in
  let before = Nx.to_array rates in
  ignore (call ~rates ~spikes ());
  let after = Nx.to_array rates in
  if not (Array.equal Float.equal before after)
  then failwith "co_smoothing mutated rates_heldout"

let () =
  (* An explicit mask derived from NaN spikes must agree with the NaN path. *)
  let rates = load_float "nan_spikes_rates" in
  let spikes = load_float "nan_spikes_spikes" in
  let mask = load_bool "nan_spikes_mask" in
  let without = call ~rates ~spikes () in
  let with_mask = call ~rates ~spikes ~mask () in
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
  expect_invalid "rank" (fun () -> call ~rates:rates_2d ~spikes:spikes_2d ());
  expect_invalid "shape" (fun () -> call ~rates:rates_3d ~spikes:spikes_wrong_neurons ());
  expect_invalid "mask shape" (fun () ->
    call ~rates:rates_3d ~spikes:spikes_3d ~mask:mask_wrong_neurons ());
  Printf.printf
    "bits: max relative error=%.3e max absolute error (expected zero)=%.3e\n%!"
    !max_relative
    !max_absolute_zero
