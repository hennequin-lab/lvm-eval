(** Internal coefficient of determination. *)

(** [score ~y_true ~y_pred] is the mean R² over the columns of two
    [row;neuron] tensors, using the reference's force-finite rule: a constant
    target scores 1 when predicted exactly and 0 otherwise.

    It raises [Invalid_argument] unless both tensors have at least two rows and
    one neuron, and otherwise share a shape. *)
val score : y_true:Nx.float64_t -> y_pred:Nx.float64_t -> float
