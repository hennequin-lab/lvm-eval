(** Internal centered ridge regression with contiguous K-fold selection. *)

(** A fitted ridge map: centered features, centered targets, and coefficients. *)
type t

(** The reference grid: nine log-spaced values from 1e-4 to 1. *)
val default_alphas : float array

(** [fit ~x ~y ~alpha] fits a ridge map with an unpenalized intercept.

    [x] and [y] are [sample;feature] matrices with the same sample count;
    [alpha] must be finite and positive. Non-finite inputs and empty feature
    dimensions raise [Invalid_argument]. *)
val fit : x:Nx.float64_t -> y:Nx.float64_t -> alpha:float -> t

(** [predict m x] applies the fitted map to [sample;feature] rows. *)
val predict : t -> Nx.float64_t -> Nx.float64_t

(** [select_alpha ~x ~y ~alphas ~folds] selects the alpha with the highest
    mean validation R² over contiguous folds of the flattened samples. Ties
    keep the earlier alpha.

    Raises [Invalid_argument] unless [folds >= 2], the alpha grid is non-empty,
    and there are at least two samples per fold ([samples >= 2 * folds]). *)
val select_alpha
  :  x:Nx.float64_t
  -> y:Nx.float64_t
  -> alphas:float array
  -> folds:int
  -> float
