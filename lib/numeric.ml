let zero_rate_floor = 1e-9
let log2 = Float.log 2.
let floor_zero_rates x = Nx.where (Nx.equal_s x 0.) (Nx.full_like x zero_rate_floor) x
