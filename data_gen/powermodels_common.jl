using PowerModels
using Ipopt
using JSON3
using JuMP

PowerModels.silence()

function compact_result(result)
    solution = get(result, "solution", Dict{String, Any}())
    bus_payload = Dict{String, Any}()
    for (index, row) in get(solution, "bus", Dict{String, Any}())
        vm = get(row, "vm", NaN)
        va = get(row, "va", 0.0)
        if isfinite(vm) && isfinite(va)
            bus_payload[string(index)] = Dict("vm" => vm, "va" => va)
        end
    end
    gen_payload = Dict{String, Any}()
    for (index, row) in get(solution, "gen", Dict{String, Any}())
        pg = get(row, "pg", NaN)
        qg = get(row, "qg", NaN)
        if isfinite(pg) && isfinite(qg)
            gen_payload[string(index)] = Dict("pg" => pg, "qg" => qg)
        end
    end
    payload = Dict{String, Any}(
        "termination_status" => string(get(result, "termination_status", "unknown")),
        "solution" => Dict("bus" => bus_payload, "gen" => gen_payload),
    )
    objective = get(result, "objective", NaN)
    if isfinite(objective)
        payload["objective"] = objective
    end
    return payload
end

function solve_to_json(input_path, output_path; max_iter=3000)
    network = PowerModels.parse_file(input_path)
    result = PowerModels.solve_ac_opf(
        network,
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "tol" => 1e-8,
            "constr_viol_tol" => 1e-8,
            "max_iter" => max_iter,
            "print_level" => 0,
            "sb" => "yes",
        ),
    )
    open(output_path, "w") do io
        JSON3.write(io, compact_result(result))
    end
end
