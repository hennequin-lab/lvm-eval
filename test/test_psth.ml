open Nx

let rec find_root dir =
  if Sys.file_exists (Filename.concat dir "test/fixtures/psth/expected.txt")
  then dir
  else (
    let parent = Filename.dirname dir in
    if String.equal parent dir
    then failwith "test_psth: cannot find test/fixtures/psth/expected.txt"
    else find_root parent)

let root = find_root (Sys.getcwd ())
let fixture_dir = Filename.concat root "test/fixtures/psth"
let input name = Filename.concat fixture_dir ("inputs/" ^ name ^ ".npy")
let load_float name = Nx_io.load_npy (input name) |> Nx.unpack Nx.float64

let load_int name =
  Nx_io.load_npy (input name)
  |> Nx.unpack Nx.int64
  |> Nx.to_array
  |> Array.map Int64.to_int

let load_cond_idx name =
  let flat = load_int (name ^ "_cond_idx") in
  let sizes = load_int (name ^ "_cond_sizes") in
  let conditions = Array.length sizes in
  let width = if Int.equal conditions 0 then 0 else Array.length flat / conditions in
  Array.init conditions (fun condition ->
    Array.init sizes.(condition) (fun index -> flat.((condition * width) + index)))

let load_jitter name =
  if Sys.file_exists (input (name ^ "_jitter"))
  then Some (load_int (name ^ "_jitter"))
  else None

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
    | _ -> failwith ("test_psth: malformed expected line: " ^ line))

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

let check_unchanged_int name before after =
  if not (Array.equal Int.equal before after)
  then failwith (name ^ " mutated an integer input")

let check_unchanged_cond name before after =
  if not (Array.equal (Array.equal Int.equal) before after)
  then failwith (name ^ " mutated eval_cond_idx")

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

let call ~rates ~psth ~cond_idx ?jitter () =
  Lvm_eval.Psth.r2 ~rates_eval:rates ~psth ~eval_cond_idx:cond_idx ?jitter ()

let () =
  List.iter
    (fun (name, expected_text) ->
       let rates = load_float (name ^ "_rates") in
       let psth = load_float (name ^ "_psth") in
       let cond_idx = load_cond_idx name in
       let jitter = load_jitter name in
       if String.equal expected_text "error"
       then expect_invalid name (fun () -> call ~rates ~psth ~cond_idx ?jitter ())
       else (
         let expected = float_of_string expected_text in
         let got = call ~rates ~psth ~cond_idx ?jitter () in
         check_value name expected got))
    expected_cases

let () =
  let rates = load_float "normal_rates" in
  let psth = load_float "normal_psth" in
  let cond_idx = load_cond_idx "normal" in
  let rates_before = Nx.to_array rates in
  let psth_before = Nx.to_array psth in
  let cond_before = Array.map Array.copy cond_idx in
  ignore (call ~rates ~psth ~cond_idx ());
  if not (Array.equal Float.equal rates_before (Nx.to_array rates))
  then failwith "Psth.r2 mutated rates_eval";
  if not (Array.equal Float.equal psth_before (Nx.to_array psth))
  then failwith "Psth.r2 mutated psth";
  check_unchanged_cond "Psth.r2 eval_cond_idx" cond_before cond_idx

let () =
  let rates = load_float "jitter_rates" in
  let psth = load_float "jitter_psth" in
  let cond_idx = load_cond_idx "jitter" in
  let jitter = load_jitter "jitter" |> Option.get in
  let rates_before = Nx.to_array rates in
  let psth_before = Nx.to_array psth in
  let cond_before = Array.map Array.copy cond_idx in
  let jitter_before = Array.copy jitter in
  ignore (call ~rates ~psth ~cond_idx ~jitter ());
  if not (Array.equal Float.equal rates_before (Nx.to_array rates))
  then failwith "Psth.r2 mutated jitter rates_eval";
  if not (Array.equal Float.equal psth_before (Nx.to_array psth))
  then failwith "Psth.r2 mutated jitter psth";
  check_unchanged_cond "Psth.r2 jitter eval_cond_idx" cond_before cond_idx;
  check_unchanged_int "Psth.r2 jitter" jitter_before jitter

let () =
  let rates = load_float "normal_rates" in
  let psth = load_float "normal_psth" in
  let cond_idx = load_cond_idx "normal" in
  let rates_2d =
    create
      float64
      [| 3; 5 |]
      [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14. |]
  in
  let psth_2d = create float64 [| 5; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10. |] in
  let rates_wrong_time = create float64 [| 3; 4; 2 |] (Array.make 24 1.) in
  let rates_wrong_neurons = create float64 [| 3; 5; 3 |] (Array.make 45 1.) in
  let bad_padding = Nx.to_array psth |> Array.copy in
  bad_padding.((2 * 2) + 1) <- Float.nan;
  let bad_padding = create float64 (Nx.shape psth) bad_padding in
  expect_invalid_contains
    "rates rank"
    [ "Lvm_eval.Psth.r2"; "rates_eval"; "[trial;time;neuron]" ]
    (fun () -> call ~rates:rates_2d ~psth ~cond_idx ());
  expect_invalid_contains
    "psth rank"
    [ "Lvm_eval.Psth.r2"; "psth"; "[condition;time;neuron]" ]
    (fun () -> call ~rates ~psth:psth_2d ~cond_idx ());
  expect_invalid_contains "time" [ "Lvm_eval.Psth.r2"; "time length" ] (fun () ->
    call ~rates:rates_wrong_time ~psth ~cond_idx ());
  expect_invalid_contains "neurons" [ "Lvm_eval.Psth.r2"; "neuron count" ] (fun () ->
    call ~rates:rates_wrong_neurons ~psth ~cond_idx ());
  expect_invalid_contains "conditions" [ "Lvm_eval.Psth.r2"; "eval_cond_idx" ] (fun () ->
    call ~rates ~psth ~cond_idx:[| [| 0 |] |] ());
  expect_invalid_contains "jitter length" [ "Lvm_eval.Psth.r2"; "jitter" ] (fun () ->
    call ~rates ~psth ~cond_idx ~jitter:[| 0 |] ());
  expect_invalid_contains "index" [ "Lvm_eval.Psth.r2"; "trial index" ] (fun () ->
    call ~rates ~psth ~cond_idx:[| [| 0; 3 |]; [| 2 |] |] ());
  expect_invalid_contains
    "first-neuron padding"
    [ "Lvm_eval.Psth.r2"; "psth contains NaN" ]
    (fun () -> call ~rates ~psth:bad_padding ~cond_idx ())

let () =
  let rates = Nx.full Nx.float64 [| 1; 5; 2 |] 1. in
  let psth =
    create
      float64
      [| 1; 5; 2 |]
      [| 1.
       ; 2.
       ; Float.nan
       ; Float.nan
       ; Float.nan
       ; Float.nan
       ; Float.nan
       ; Float.nan
       ; Float.nan
       ; Float.nan
      |]
  in
  expect_invalid_contains
    "one valid PSTH row"
    [ "Lvm_eval.R2.score"; "at least two rows" ]
    (fun () -> call ~rates ~psth ~cond_idx:[| [| 0 |] |] ())

let () =
  let psth_constant = Nx.full Nx.float64 [| 2; 5; 2 |] 3. in
  let rates_constant = Nx.full Nx.float64 [| 3; 5; 2 |] 3. in
  let cond_idx = [| [| 0; 1 |]; [| 2 |] |] in
  let perfect = call ~rates:rates_constant ~psth:psth_constant ~cond_idx () in
  let imperfect =
    call ~rates:(Nx.full Nx.float64 [| 3; 5; 2 |] 4.) ~psth:psth_constant ~cond_idx ()
  in
  check_value "constant-perfect" 1. perfect;
  check_value "constant-imperfect" 0. imperfect;
  Printf.printf
    "psth: max relative error=%.3e max absolute error (expected zero)=%.3e\n%!"
    !max_relative
    !max_absolute_zero
