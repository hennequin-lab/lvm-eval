open Nx

let rec find_root dir =
  if Sys.file_exists (Filename.concat dir "test/fixtures/dmfc_code/expected.txt")
  then dir
  else (
    let parent = Filename.dirname dir in
    if String.equal parent dir
    then failwith "test_dmfc: cannot find test/fixtures/dmfc_code/expected.txt"
    else find_root parent)

let root = find_root (Sys.getcwd ())
let fixture_dir = Filename.concat root "test/fixtures/dmfc_code"
let input name = Filename.concat fixture_dir ("inputs/" ^ name ^ ".npy")
let load_float name = Nx_io.load_npy (input name) |> Nx.unpack Nx.float64
let load_bool name = Nx_io.load_npy (input name) |> Nx.unpack Nx.bool

let load_int name =
  Nx_io.load_npy (input name)
  |> Nx.unpack Nx.int64
  |> Nx.to_array
  |> Array.map Int64.to_int

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
    | _ -> failwith ("test_dmfc: malformed expected line: " ^ line))

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

let check_value name expected got =
  let error = relative_error expected got in
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

let call ~rates ~mask ~conditions ~tp () =
  Lvm_eval.Dmfc.speed_tp_code ~rates_eval:rates ~mask ~conditions ~tp ()

let () =
  List.iter
    (fun (name, expected_text) ->
       let rates = load_float (name ^ "_rates") in
       let mask = load_bool (name ^ "_mask") in
       let conditions = load_int (name ^ "_conditions") in
       let tp =
         Nx_io.load_npy (input (name ^ "_tp")) |> Nx.unpack Nx.float64 |> Nx.to_array
       in
       if String.equal expected_text "error"
       then
         if String.equal name "singleton_condition"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.Dmfc.speed_tp_code"; "speed"; "two samples" ]
             (fun () -> call ~rates ~mask ~conditions ~tp ())
         else if String.equal name "constant_condition"
         then
           expect_invalid_contains
             name
             [ "Lvm_eval.Dmfc.speed_tp_code"; "constant" ]
             (fun () -> call ~rates ~mask ~conditions ~tp ())
         else
           expect_invalid_contains
             name
             [ "Lvm_eval.Dmfc.speed_tp_code"; "mask"; "time point" ]
             (fun () -> call ~rates ~mask ~conditions ~tp ())
       else (
         let expected = float_of_string expected_text in
         let got = call ~rates ~mask ~conditions ~tp () in
         check_value name expected got))
    expected_cases

let () =
  let rates = load_float "normal_rates" in
  let mask = load_bool "normal_mask" in
  let conditions = load_int "normal_conditions" in
  let tp = Nx_io.load_npy (input "normal_tp") |> Nx.unpack Nx.float64 |> Nx.to_array in
  let rates_before = Nx.to_array rates in
  let mask_before = Nx.to_array mask in
  let conditions_before = Array.copy conditions in
  let tp_before = Array.copy tp in
  ignore (call ~rates ~mask ~conditions ~tp ());
  if not (Array.equal Float.equal rates_before (Nx.to_array rates))
  then failwith "Dmfc.speed_tp_code mutated rates_eval";
  if not (Array.equal Bool.equal mask_before (Nx.to_array mask))
  then failwith "Dmfc.speed_tp_code mutated mask";
  if not (Array.equal Int.equal conditions_before conditions)
  then failwith "Dmfc.speed_tp_code mutated conditions";
  if not (Array.equal Float.equal tp_before tp)
  then failwith "Dmfc.speed_tp_code mutated tp"

let () =
  let rates = load_float "normal_rates" in
  let mask = load_bool "normal_mask" in
  let conditions = load_int "normal_conditions" in
  let tp = Nx_io.load_npy (input "normal_tp") |> Nx.unpack Nx.float64 |> Nx.to_array in
  let rates_2d = create float64 [| 4; 5 |] (Array.make 20 1.) in
  let mask_3d = create bool [| 4; 5; 1 |] (Array.make 20 true) in
  let mask_wrong = create bool [| 4; 4 |] (Array.make 16 true) in
  expect_invalid_contains
    "rates rank"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "rates_eval"; "[trial;time;neuron]" ]
    (fun () -> call ~rates:rates_2d ~mask ~conditions ~tp ());
  expect_invalid_contains
    "mask rank"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "mask"; "[row;neuron]" ]
    (fun () -> call ~rates ~mask:mask_3d ~conditions ~tp ());
  expect_invalid_contains
    "mask shape"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "mask"; "shape" ]
    (fun () -> call ~rates ~mask:mask_wrong ~conditions ~tp ());
  expect_invalid_contains
    "conditions length"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "conditions"; "entries" ]
    (fun () -> call ~rates ~mask ~conditions:[| 0; 1 |] ~tp ());
  expect_invalid_contains
    "tp length"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "tp"; "entries" ]
    (fun () -> call ~rates ~mask ~conditions ~tp:[| 0.1 |] ());
  expect_invalid_contains
    "mask no time"
    [ "Lvm_eval.Dmfc.speed_tp_code"; "mask"; "time point" ]
    (fun () ->
       let zero_mask = create bool [| 4; 5 |] (Array.make 20 false) in
       call ~rates ~mask:zero_mask ~conditions ~tp ())
