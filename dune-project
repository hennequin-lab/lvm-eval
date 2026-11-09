(lang dune 3.24)

(name lvm-eval)

(license ISC)

(authors "The Hennequin Lab")

(implicit_transitive_deps false)

; Murder is developed beside this project, so pin the working tree by path.
; Dune resolves the pins declared in murder's dune-project recursively, so the
; Raven commit comes from murder's own pin. `cmdargs` is only the recursive-pin
; anchor; lvm-eval does not depend on it.
(pin
 (url "../murder")
 (package
  (name cmdargs)))

(package
 (name lvm-eval)
 (synopsis "Evaluation metrics for latent variable models")
 (description
  "Pure OCaml evaluation metrics for latent variable models of neural population activity. The first release implements the NLB'21 metric set on [trial; time; neuron] tensors.")
 (license ISC)
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  nx))
