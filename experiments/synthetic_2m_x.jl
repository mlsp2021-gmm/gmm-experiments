using Revise

using DrWatson
@quickactivate :GammaMixtureExperiments

using Distributions
using ReactiveMP
using Random
using Plots

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
experiments = dict_list(Dict(
    "n" => 2500,
    "iterations" => 2500,
    "epsilon" => 100.0,
    "seed" => collect(1:10),
    "constraint" => [ EM(), Marginalisation() ],
    "prod_strategy" => [
        @onlyif("constraint" == EM(), FoldRightProdStrategy()),
        @onlyif("constraint" == Marginalisation(), FoldLeftProdStrategy()),
    ],
    "meta" => [
        @onlyif("constraint" == EM(), nothing),
        @onlyif("constraint" == Marginalisation(), ImportanceSamplingApproximation(MersenneTwister(1234), 5000))
    ],
    "jitter" => 0.0,
    "shift" => 10.0
))

function run_experiment(params)
    # We unpack all provided parameters into separate variables
    @unpack n, iterations, epsilon, seed, constraint, prod_strategy, meta, jitter, shift = params

    # For reproducibility
    rng = MersenneTwister(seed)

    mixtures  = [ Gamma(9.0, inv(27.0)), Gamma(90.0, inv(270.0)) ]
    nmixtures = length(mixtures)
    mixing    = rand(rng, nmixtures)
    mixing    = mixing ./ sum(mixing)
    mixture   = MixtureModel(mixtures, mixing)

    dataset = rand(rng, mixture, n)

    # Priors are mostly vague and use information from dataset only (random means and fixed variances)
    priors_as, priors_bs = generate_priors(dataset, nmixtures, seed = seed, ϵ = epsilon, jitter = jitter, shift = shift)

    parameters = GammaMixtureModelParameters(
        nmixtures        = nmixtures,
        priors_as        = priors_as,
        priors_bs        = priors_bs,
        prior_s          = Dirichlet(10000 * mixing),
        as_prod_strategy = prod_strategy,
        as_constraint    = constraint,
        meta             = meta
    )

    result = InferenceResults(gamma_mixture_inference(dataset, iterations, parameters)...)

    # Specify which information should be saved in JLD2 file
    return @strdict result parameters mixtures mixing mixture dataset params
end

function generate_plots(input; with_pgf = true)
    try
        @unpack result, parameters, mixtures, mixing, mixture, dataset, params = input

        save_types  = (String, EM, Marginalisation, FoldLeftProdStrategy, FoldRightProdStrategy, Real, ImportanceSamplingApproximation)
        gen_folder = projectdir("results", "synthetic", "2mixtures", "with_known_mixing")
        stats_path = projectdir("results", "synthetic", "2mixtures", "with_known_mixing", savename("stats", params, allowedtypes = save_types))

        mkpath(gen_folder)

        # Save stats file, do we really need it?
        open(io -> stats(io, result), stats_path, "w")

        if with_pgf
            pgf_path = projectdir("results", "synthetic", "tikz", "2mixtures", "with_known_mixing", savename("fig", params, "tikz", allowedtypes = save_types))
            pgf_densities(string(params["constraint"]), mixture, result, dataset, pgf_path)
        end

        fig_path = projectdir("results", "synthetic", "2mixtures", "with_known_mixing", savename("fig", params, "png", allowedtypes = save_types))

        savefig(compare(mixture, result), fig_path)
    catch error
        @warn "Could not save plots: $error"
    end
end

results = map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "synthetic", "2mixtures", "with_known_mixing")
    # Types which should be used for cache file name
    save_types  = (String, EM, Marginalisation, FoldLeftProdStrategy, FoldRightProdStrategy, Real, ImportanceSamplingApproximation)
    try
        result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types, tag = false) do params
            run_experiment(params)
        end

        generate_plots(result, with_pgf = false)

        return result
    catch error
        @warn "Error during experiment $error"
        return error
    end
end;

begin
    em_id = 10
    vi_id = em_id + div(length(results), 2)
    em_result = results[em_id]["result"]
    vi_result = results[vi_id]["result"]
    dataset = results[vi_id]["dataset"]
    pgf_path = projectdir("results", "synthetic", "tikz", "2mixtures", "comparison", savename("comaprison_2_mixing_seed_$(em_id)", [], "tikz")) 
    pgf_mixture(em_result, vi_result, dataset, pgf_path)
end
