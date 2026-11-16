(** Internal float64 numeric helpers. *)

(** The reference's replacement for exact zero rates. *)
val zero_rate_floor : float

(** [Float.log 2.]. *)
val log2 : float

(** [floor_zero_rates x] replaces exact zeros with {!zero_rate_floor} on a
    copy; the input is not mutated. *)
val floor_zero_rates : Nx.host Nx.float64_t -> Nx.host Nx.float64_t
