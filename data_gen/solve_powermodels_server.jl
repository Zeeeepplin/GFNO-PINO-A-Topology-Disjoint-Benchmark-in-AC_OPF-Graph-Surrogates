include("powermodels_common.jl")

max_iter = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000

println("__TOPOLOGY_PINO_READY__")
flush(stdout)

for command in eachline(stdin)
    if command == "__QUIT__"
        break
    end
    fields = split(command, '\t'; limit=2)
    if length(fields) != 2
        println("__TOPOLOGY_PINO_ERROR__")
        flush(stdout)
        continue
    end
    try
        solve_to_json(String(fields[1]), String(fields[2]); max_iter=max_iter)
        println("__TOPOLOGY_PINO_OK__")
    catch exception
        message = sprint(showerror, exception, catch_backtrace())
        open(String(fields[2]), "w") do io
            JSON3.write(
                io,
                Dict(
                    "termination_status" => "ERROR",
                    "error" => message,
                    "solution" => Dict("bus" => Dict(), "gen" => Dict()),
                ),
            )
        end
        println("__TOPOLOGY_PINO_ERROR__")
    end
    flush(stdout)
end
