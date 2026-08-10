include("powermodels_common.jl")

if length(ARGS) < 2 || length(ARGS) > 3
    error("usage: solve_powermodels.jl INPUT.m OUTPUT.json [MAX_ITER]")
end

max_iter = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 3000
solve_to_json(ARGS[1], ARGS[2]; max_iter=max_iter)
