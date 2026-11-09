open Nx

let () =
  let x = create float64 [| 3 |] [| 1.; 2.; 3. |] in
  let total = item [] (sum x) in
  assert (Float.equal total 6.)
