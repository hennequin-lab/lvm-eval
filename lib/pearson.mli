(** Internal sample Pearson correlation. *)

(** [score ~fn ~x_arg ~y_arg ~x ~y] is the sample Pearson r of two equal-length
    rank-1 tensors. It raises [Invalid_argument] for fewer than two entries,
    non-finite values, or zero variance. *)
val score
  :  fn:string
  -> x_arg:string
  -> y_arg:string
  -> x:Nx.host Nx.float64_t
  -> y:Nx.host Nx.float64_t
  -> float
