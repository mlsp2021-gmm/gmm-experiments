using Revise

using DrWatson
@quickactivate :GammaMixtureExperiments

using Distributions
using ReactiveMP
using Random
using Plots
using BenchmarkTools
using ProgressMeter

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
# In this experiment we do not really care about our inference results but only about performance on average between EM and MM
# We check inference results in a different experiments
experiments = dict_list(Dict(
    "n" => 250,
    "iterations" => 25,
    "epsilon" => 100.0,
    "seed" => 1,
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
    "shift" => 10.0,
    "mindex" => [ 1, 2, 3 ],
    "benchmark" => true
))

function run_benchmark_experiment(params)
    # We unpack all provided parameters into separate variables
    @unpack n, iterations, epsilon, seed, constraint, prod_strategy, meta, jitter, shift, mindex, benchmark = params

    @assert benchmark "This experiment assumes benchmark parameter to be set to true"
    @assert 1 ≤ mindex ≤ 3 "Invalid mindex: $mindex"

    # For reproducibility
    rng = MersenneTwister(seed)

    mixtures_candidates = [
        [ Gamma(9.0, inv(27.0)), Gamma(90.0, inv(270.0)) ], 
        [ Gamma(40.0, inv(20.0)), Gamma(6.0, inv(1.0)), Gamma(200.0, inv(20.0)) ],
        [ Gamma(200.0, inv(100.0)), Gamma(400.0, inv(100.0)), Gamma(600.0, inv(100.0)), Gamma(800.0, inv(100.0)) ]
    ]

    mixtures  = mixtures_candidates[mindex]
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

    result    = InferenceResults(gamma_mixture_inference(dataset, iterations, parameters; with_progress = false)...)
    benchmark = @benchmark gamma_mixture_inference($dataset, $iterations, $parameters; with_progress = false)

    # Specify which information should be saved in JLD2 file
    return @strdict benchmark result parameters mixtures mixing mixture dataset params
end

results = ProgressMeter.@showprogress map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "synthetic", "benchmark", "with_known_mixing")
    # Types which should be used for cache file name
    save_types  = (String, EM, Marginalisation, FoldLeftProdStrategy, FoldRightProdStrategy, Real, ImportanceSamplingApproximation)
    result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types, tag = false) do params
        run_benchmark_experiment(params)
    end
    return result
end;

table = collect_results(projectdir("dump", "synthetic", "benchmark", "with_known_mixing"))

function compare_mindex(table, mindex)
    vi = filter(r -> r["params"]["constraint"] == Marginalisation() && r["params"]["mindex"] == mindex, table)[!, "benchmark"]
    em = filter(r -> r["params"]["constraint"] == ExpectationMaximisation() && r["params"]["mindex"] == mindex, table)[!, "benchmark"]
    
    et = (t) -> mean(t).time

    return sum(et.(vi)) / sum(et.(em))
end

function compare(table)
    l = 3
    r = 0.0
    for i in 1:l
        r += compare_mindex(table, i)
    end
    return r ./ l
end

compare_mindex(table, 1) # 27
compare_mindex(table, 2) # 30
compare_mindex(table, 3) # 35
compare(table) # 31