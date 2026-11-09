(** Internal valid-entry masks and masked reductions. *)

(** [valid ~fn ?mask spikes] is [mask] when supplied; otherwise it is the
    entries of [spikes] that are not NaN. *)
val valid : fn:string -> ?mask:Nx.bool_t -> Nx.float64_t -> Nx.bool_t

(** [selected mask x] is [x] where [mask] is true and zero elsewhere. *)
val selected : Nx.bool_t -> Nx.float64_t -> Nx.float64_t

(** [count_per_neuron mask] sums [mask] over trial and time, leaving one count
    per neuron. *)
val count_per_neuron : Nx.bool_t -> Nx.float64_t

(** [sum_per_neuron mask x] sums [x] over trial and time where [mask] is true,
    leaving one sum per neuron. *)
val sum_per_neuron : Nx.bool_t -> Nx.float64_t -> Nx.float64_t

(** [total mask x] sums [x] over all entries where [mask] is true. *)
val total : Nx.bool_t -> Nx.float64_t -> Nx.float64_t
