let valid ~fn ?mask spikes =
  ignore fn;
  match mask with
  | Some mask -> mask
  | None -> Nx.logical_not (Nx.isnan spikes)

let selected mask x = Nx.where mask x (Nx.zeros_like x)
let count_per_neuron mask = Nx.sum ~axes:[ 0; 1 ] (Nx.cast Nx.float64 mask)
let sum_per_neuron mask x = Nx.sum ~axes:[ 0; 1 ] (selected mask x)
let total mask x = Nx.sum (selected mask x)
