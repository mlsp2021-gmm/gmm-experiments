export infgamma, generate_random_priors, generate_priors
export InferenceResults, stats, compare, pgf_histogram, pgf_densities, pgf_mixture

using Distributions
using Random
using Plots
using StatsPlots
using PGFPlotsX
using ColorSchemes
using StatsBase: Histogram, fit, normalize

# helper function for gamma distribution
infgamma(T, x; ϵ = 1e-3) = GammaShapeRate{T}(x^2 / ϵ, x / ϵ)

# confusion matrix
function confusionmatrix(predictions::AbstractArray{T}, references::AbstractArray{T}; showmatrix::Bool=false, normalize::Bool=false, positive=nothing) where T
    @assert length(predictions) == length(references) "[confusionmatrix] The predicted and reference labels should have the same length."

    # If positive class is specified, binarize labels
    _binarize_(predictions,references) =
    predictions, references, sort(unique(predictions)), sort(unique(references))

    function _binarize_(predictions::AbstractArray{T}, references::AbstractArray{T}, positive::T)
        @assert positive in references "[confusionmatrix] $(positive) is not in the reference label vector."
        yb = falses(length(predictions))
        yrb = falses(length(references))
        yb[predictions.==positive] .= true
        yrb[references.==positive] .= true
        return yb, yrb, sort(unique(yb),rev=true), sort(unique(yrb),rev=true)
    end
    y, yr, yu, yru = _binarize_(predictions, references, positive)


    # Construct confusion matrix
    C = length(yru)
    @assert issubset(yu,yru) "[confusionmatrix] The predicted labels should be a subset of the reference labels."
    cm = zeros(Int64, C,C)
    @inbounds for (j,cr) in enumerate(yru)
        for (i,cp) in enumerate(yru)
            cm[i,j] = sum((yr .== cr) .& (y .== cp))
        end
    end

    # Check for normalization
    if normalize
        # Loop through the classes and normalize the columns
        # of the confusion matrix with respect to their sum
        # i.e. the sum of each column should be 1.0
        for j in 1:C
            cm[:,j]/=sum(yr .==yru[j])
        end
    end

    # Check if the matrix should be nicely printed or not
    if showmatrix
        println()
        println("reference labels (columns):")
    end
    lsize=ceil(Int, log10(length(y)))+2
    println(sprint((io::IO,v)->map(x->print(io,lpad(" \"$x\" ",lsize)),v), yru))

    println(repeat("-", (lsize+3)*C))
    for i in 1:size(cm,1)
        for j in 1:size(cm,2)
            print(lpad("$(cm[i,j])   ",lsize))
        end
        println()
    end
    println(repeat("-", (lsize+3)*C))

    return cm
end

# function for calculating the accuracy of a binomial classification problem
function accuracy(y_pred, y_true)
    @assert length(unique(y_pred)) == 2 "Only binomial clusters are supported"
    map((x) -> x < 50 ? 100-x : x, sum(y_true .== y_pred) /length(y_true)*100)
end

# plot function for gamma
function plot_gamma(ax, data, a, b, w, color, linewidth)

    # create support
    x = minimum(data) : (maximum(data) - minimum(data))/1000 : maximum(data)

    # calculate gamma distribution
    y = w .* pdf(Gamma(a, 1/b), x)

    # plot function
    ax.plot(x, y, color=color, linewidth=linewidth)

end

function generate_priors(dataset::AbstractMatrix, nmixtures; jitter = 1e-3, seed = 123)
    (nobservations, nfeatures) = size(dataset)

    rng = MersenneTwister(seed)

    prior_m = repeat(mean(dataset, dims = 1)[:], inner = nmixtures)
    prior_v = repeat(var(dataset, dims = 1)[:], inner = nmixtures)

    priors_as = Vector(undef, nfeatures * nmixtures)
    priors_bs = Vector(undef, nfeatures * nmixtures)

    for i in 1:nmixtures
        for j in 1:nfeatures
            k = i + (j - 1) * nmixtures
            priors_as[k] = infgamma(Float64, prior_m[k] ^ 2 / prior_v[k] * (1 + jitter * rand(rng)))
            priors_bs[k] = infgamma(Float64, prior_m[k] / prior_v[k] * (1 + jitter * rand(rng)))
        end
    end

    return priors_as, priors_bs
end

function generate_priors(dataset::AbstractVector, nmixtures; ϵ = 1000.0, jitter = 0.0, shift = 0.0, seed = 123)

    rng = MersenneTwister(seed)

    prior_ms = rand(rng, dataset, nmixtures)
    prior_vs = ones(nmixtures)

    priors_as = Vector(undef, nmixtures)
    priors_bs = Vector(undef, nmixtures)

    for i in 1:nmixtures
        priors_as[i] = infgamma(Float64, shift + prior_ms[i] ^ 2 / prior_vs[i] * (1 + jitter * rand(rng)), ϵ = ϵ)
        priors_bs[i] = infgamma(Float64, shift + prior_ms[i] / prior_vs[i] * (1 + jitter * rand(rng)), ϵ = ϵ)
    end

    return priors_as, priors_bs
end


function generate_random_priors(nmixtures; jitter = 10.0, seed = 123)
    rng = MersenneTwister(seed)

    priors_as    = map(p -> infgamma(Float64, p, ϵ = 100.0), sqrt(jitter) * rand(rng, nmixtures))
    priors_bs    = map(p -> infgamma(Float64, p, ϵ = 100.0), sqrt(jitter) * rand(rng, nmixtures))

    return priors_as, priors_bs
end

# Estimated Results and helper functions

struct InferenceResults
    mswitch
    mselector
    mas
    mbs
    fe
end

mixing(result::InferenceResults) = mean(result.mswitch[end])
means(result::InferenceResults)  = (result.mas |> last .|> mean) ./ (result.mbs |> last .|> mean)

function stats(results::InferenceResults)
    stats(stdout, results)
end

function stats(io::IO, results::InferenceResults)
    println(io, "Mixing: ", mixing(results))
    println(io, "Means:  ", means(results))
    println(io, "FE:     ", last(results.fe))
end

mixing(mm::MixtureModel) = mm.prior.p
mixing(mm::Gamma) = [ 1.0 ]

function compare(real, result)
    p1 = histogram(rand(real, 10000), bins = 100, label = "real", normalize = :pdf)
    p1 = plot!(p1, real, linewidth = 2, label = false)

    _mixing = mixing(result)

    _estimated_dists   = map(g -> Gamma(g[1], inv(g[2])), zip(mean.(result.mas[end]), mean.(result.mbs[end])))
    _estimated_mixture = MixtureModel(_estimated_dists, _mixing)

    p2 = histogram(rand(_estimated_mixture, 10000), bins = 100, label = "estimated", normalize = :pdf)
    p2 = plot!(p2, _estimated_mixture, linewidth = 2, label = false)

    p3 = plot(result.fe, legend = false)
    p4 = plot(result.fe[Int(round(length(result.fe) * 3/4)):end], legend = false)

    p5 = plot(sort(mixing(real)), seriestype = :bar, legend = false)
    p6 = plot(sort(_mixing), seriestype = :bar, legend = false)

    plot(p1, p2, p5, p6, p3, p4, layout = @layout([ a b; c d; e f ]), size = (800, 800))
end

# Plot validation results
function pgf_histogram(inf_type::String, mixing::Vector{Float64}, mas::Vector{Float64}, mbs::Vector{Float64}, dataset)
    _dists   = map(g -> Gamma(g[1], inv(g[2])), zip(mas, mbs))
    _mixture = MixtureModel(_dists, mixing)


    h = fit(Histogram, rand(_mixture, 10000), nbins=100, closed = :left)
    h = normalize(h, mode=:pdf)

    nmixtures = length(mixing)

    x = range(0, maximum(dataset); length = 100)
    the_title = inf_type !== "" ? "Histogram of recovered $(nmixtures) mixtures ($inf_type)" : "Histogram of $(nmixtures) mixtures"
    plt_mixture = @pgf Axis(
        {
            title=the_title,
            yticklabel_style={
            "/pgf/number format/fixed,
            /pgf/number format/precision=3"
            },
            grid="major",
            yminorgrids=true,
            tick_align="outside",
            xtick=range(0, maximum(dataset); length=5),
            xmin =0.0,
            xmax=maximum(dataset),
            scaled_y_ticks = false,
        },
        Plot({"ybar interval", fill="blue!15", "forget plot"}, Table(h)),
        Plot({style="{thick, dashed}", color="magenta"}, Table(x, pdf.(_mixture, x))), LegendEntry("Mixture"),
        Iterators.flatten([
            [Plot({style="{thick}", color=col}, Table(x, mixing[i]*pdf.(_dists[i], x))), LegendEntry("$i component")] for (col, i) in zip(colorschemes[:seaborn_bright], 1:nmixtures)
    ])...)
    pgfsave(projectdir("results", "synthetic", "tikz")*"/$(inf_type)_plt_mixture_$(nmixtures).tikz", plt_mixture)
    return plt_mixture
end

# Plots separate densities for verification
function pgf_densities(inf_type::String, real, result, dataset, path)

    _as, _bs = mean.(result.mas[end]), mean.(result.mbs[end])

    _dists   = map(g -> Gamma(g[1], inv(g[2])), zip(_as, _bs))
    _mixing = mean(result.mswitch[end])

    h = fit(Histogram, dataset, nbins=100, closed = :left)
    h = normalize(h, mode=:pdf)

    nmixtures = ncomponents(real)

    x = range(0, maximum(dataset); length = 100)
    the_title = "$inf_type result for $nmixtures components"
    plt_mixture = @pgf Axis(
        {
            title=the_title,
            yticklabel_style={
            "/pgf/number format/fixed,
            /pgf/number format/precision=3"},
            grid="major",
            yminorgrids=true,
            tick_align="outside",
            xtick=range(0, maximum(dataset); length=5),
            xmin =0.0,
            xmax=maximum(dataset),
            scaled_y_ticks = false,
        },
        Plot({"ybar interval", fill="blue!15", "forget plot", opacity=0.6, draw="none", "bar width"=0.000005}, Table(h)),
        Iterators.flatten([
        [Plot({style="{ultra thick}", color=col}, Table(x, _mixing[i]*pdf.(_dists[i], x))), LegendEntry("$i component")] for (col, i) in zip(colorschemes[:seaborn_bright], 1:nmixtures)
        ])...,
        Iterators.flatten([
        [Plot({style="{ultra thick, dashed}", color=col}, Table(x, probs(real)[i]*pdf.(component(real, i), x)))] for (col, i) in zip(colorschemes[:seaborn_bright], 1:nmixtures)
        ])...
    )

    pgfsave(path, plt_mixture)

    return plt_mixture
end


# Plots separate densities for verification
function pgf_mixture(result_em, result_vi, dataset, path)

    function result_to_mixture(result)
        _as, _bs = mean.(result.mas[end]), mean.(result.mbs[end])
        _dists   = map(g -> Gamma(g[1], inv(g[2])), zip(_as, _bs))
        _mixing  = mean(result.mswitch[end])
        _mixture = MixtureModel(_dists, _mixing)
    end

    vi_mixture, em_mixture   = result_to_mixture.([result_vi, result_em])

    h = fit(Histogram, dataset, nbins=100, closed = :left)
    h = normalize(h, mode=:pdf)

    nmixtures = ncomponents(vi_mixture)

    x = range(0, maximum(dataset); length = 100)
    the_title = "Inference result for M=$nmixtures"
    plt_mixture = @pgf Axis(
        {
            legend_style={"nodes={scale=0.7, transform shape}"},
            legend_cell_align="left",
            title=the_title,
            yticklabel_style={
            "/pgf/number format/fixed,
            /pgf/number format/precision=3"},
            grid="major",
            yminorgrids=true,
            tick_align="outside",
            xtick=range(0, maximum(dataset); length=5),
            xmin =0.0,
            xmax=maximum(dataset),
            scaled_y_ticks = false,
        },
        Plot({"ybar interval", fill="blue!15", "forget plot", opacity=0.6, draw="none", "bar width"=0.000005}, Table(h)),
        # Plot({style="{ultra thick, dashed}", color="magenta"}, Table(x, pdf.(real, x))), LegendEntry("Generated Mixture"),
        Plot({style="{ultra thick, dotted}", color="blue"}, Table(x, pdf.(vi_mixture, x))), LegendEntry("VI estimation"),
        Plot({style="{thick, solid}", color="green"}, Table(x, pdf.(em_mixture, x))), LegendEntry("EM estimation")
    )
    
    pgfsave(path, plt_mixture)

    return plt_mixture
end