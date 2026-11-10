(** Internal shape, placement, and precondition checks. *)

(** [require_rank2 fn arg x] checks that [x] has rank 2. *)
val require_rank2 : string -> string -> ('a, 'b) Nx.t -> unit

(** [require_rank3 fn arg x] checks that [x] has rank 3. *)
val require_rank3 : string -> string -> ('a, 'b) Nx.t -> unit

(** [require_same_shape fn arg1 arg2 x y] checks that [x] and [y] have the
    same shape. *)
val require_same_shape
  :  string
  -> string
  -> string
  -> ('a, 'b) Nx.t
  -> ('c, 'd) Nx.t
  -> unit

(** [require_host fn arg x] checks that [x] is on [Nx.Placement.host]. *)
val require_host : string -> string -> ('a, 'b) Nx.t -> unit
