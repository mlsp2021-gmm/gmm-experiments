using DrWatson
@quickactivate :GammaMixtureExperiments

using PGFPlotsX
using JLD

begin
    # setup = "with_known_shape_rate"
    setup = "with_known_mixing"
    em_fe, vi_fe = [], []
    for (root, dirs, files) in walkdir("dump")
        println("Files in $root")
        for file in files
            if occursin("em", file) && occursin(setup, root)
                em_fe = push!(em_fe, load(joinpath(root, file))["result"].fe)
            elseif occursin("vi", file) && occursin(setup, root)
                vi_fe = push!(vi_fe, load(joinpath(root, file))["result"].fe)
            end
        end
    end

    # filter runs ended with errors
    filter!(x -> isequal(length(x), length(minimum(em_fe))), em_fe)
    filter!(x -> isequal(length(x), length(minimum(vi_fe))), vi_fe)

    vi_fe_agg = sum(vi_fe) ./ length(vi_fe)
    em_fe_agg = sum(em_fe) ./ length(em_fe)

    niterations = length(vi_fe_agg)

    plt_fe = @pgf Axis({xlabel="iteration",
                        ylabel="Free energy [nats]",
                        legend_pos = "north east",
                        legend_cell_align="{left}",
                        scale = 1.0,
                        grid = "major",
                        title = "FE for synthetic dataset",
                        },
                        Plot({no_marks, color="red"}, Coordinates(collect(2:niterations), em_fe_agg[2:end])), LegendEntry("EM"),
                        Plot({no_marks, style ="{dashed}", color="blue"}, Coordinates(collect(2:niterations), vi_fe_agg[2:end])), LegendEntry("VI"))

    pgfsave("results/synthetic/tikz/plt_$(setup)_fe.tikz", plt_fe)
end

setup = "with_known_mixing"
em_fe, vi_fe = [], []
for (root, dirs, files) in walkdir("dump")
    println("Files in $root")
    for file in files
        if occursin("em", file) && occursin(setup, root)
            em_fe = push!(em_fe, load(joinpath(root, file))["result"].fe)
        elseif occursin("vi", file) && occursin(setup, root)
            vi_fe = push!(vi_fe, load(joinpath(root, file))["result"].fe)
        end
    end
end

