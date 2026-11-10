(** Internal coefficient of determination. *)

(** [score ~y_true ~y_pred] is the mean R² over the columns of two
    [row;neuron] tensors, using the reference's force-finite rule: a constant
    target scores 1 when predicted exactly and 0 otherwise. *)
val score : y_true:Nx.float64_t -> y_pred:Nx.float64_t -> float
