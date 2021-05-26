using Revise
# using Plots
using PGFPlotsX
using ReactiveMP
using DataFrames
using Random
using CSV
using UMAP
using TSne
using Distributions
using ColorSchemes
using StatsBase: Histogram, fit, normalize
using GammaMixtureExperiments

## acquire data
begin
    dataset = load_dataset(CountriesData(), Not([ :country ]); force_abs = true)

    (_, nfeatures) = size(dataset)

    nmixtures   = 5
    niterations = 100

    priors_as, priors_bs = generate_priors(dataset, nmixtures; jitter = 1e-3, seed = 5892)
end

## Sampling experiments
begin
    rng = MersenneTwister(123)

    parameters = MvGammaMixtureModelParameters(
        nmixtures        = nmixtures,
        nfeatures        = nfeatures,
        priors_as        = priors_as,
        priors_bs        = priors_bs,
        meta             = ImportanceSamplingApproximation(rng, 10000)
    )

    vi_mswitch, vi_mselector, vi_mas, vi_mbs, vi_fe = mv_gamma_mixture_inference(dataset, niterations, parameters)
end

# EM experiments
begin
    rng = MersenneTwister(123)

    parameters = MvGammaMixtureModelParameters(
        nmixtures        = nmixtures,
        nfeatures        = nfeatures,
        priors_as        = priors_as,
        priors_bs        = priors_bs,
        as_prod_strategy = FoldRightProdStrategy(),
        as_constraint    = EM(),
        meta             = EM()
    )

    em_mswitch, em_mselector, em_mas, em_mbs, em_fe = mv_gamma_mixture_inference(dataset, niterations, parameters)
end


## Free energy
# We show convergence of FE for both methods
begin
    plt_fe = @pgf Axis({xlabel="iteration",
                        ylabel="Free energy [nats]",
                        legend_pos = "north east",
                        legend_cell_align="{left}",
                        scale = 1.0,
                        grid = "major",
                        title = "FE $(nmixtures) mixtures",
                        },
                        Plot({no_marks, color="red"}, Coordinates(collect(2:niterations), em_fe[2:end])), LegendEntry("EM"),
                        Plot({no_marks, style ="{dashed}", color="blue"}, Coordinates(collect(2:niterations), vi_fe[2:end])), LegendEntry("VI"))

    # pgfsave("results/countries/plt_fe.tikz", plt_fe)
end

## Visualization of clustered data
inf_type = "EM";
# inf_type = "VI";
(mswitch, mselector, mas, mbs, fe) = inf_type == "EM" ? (em_mswitch, em_mselector, em_mas, em_mbs, em_fe) : (vi_mswitch, vi_mselector, vi_mas, vi_mbs, vi_fe)
# UMAP of classified data
begin
    label_suffix = "$inf_type $nmixtures mixtures"
    umaptY = umap(dataset'; n_neighbors=10, min_dist=0.1, n_epochs=1000)
    umapY = umaptY'
    gamma_cats = argmax.(probvec.(mselector[end]))
    map(r -> length(findall(d -> d == r, gamma_cats)), 1:nmixtures)
    plt_umap = @pgf Axis({title="UMAP "*label_suffix,
                        xlabel="projection 1",
                        ylabel="projection 2",
                        legend_pos = "north west",
                        mark_options = {scale=0.3},
                        grid="major",
                        style = {thick}
                        },
                        Plot(
                        {only_marks, scatter, scatter_src = "explicit"},
                        Table(
                        {x = "x", y = "y", meta = "col"},
                            x = umapY[:,1], y = umapY[:,2], col = gamma_cats),
                            ),
                        )
    # pgfsave("results/countries/plt_umap_$inf_type.tikz", plt_umap)
end


# tSNE of classified data
begin
    label_suffix = "$inf_type $nmixtures mixtures"
    tsneY = tsne(dataset, 2, 0, 1000, 10.0);
    gamma_cats = argmax.(probvec.(mselector[end]))
    map(r -> length(findall(d -> d == r, gamma_cats)), 1:nmixtures)
    plt_tsne = @pgf Axis({title="tSNE "*label_suffix,
                        xlabel="projection 1",
                        ylabel="projection 2",
                        legend_pos = "north west",
                        mark_options = {scale=0.3},
                        grid="major",
                        style = {thick}
                        },
                        Plot(
                        {only_marks, scatter, scatter_src = "explicit"},
                        Table(
                        {x = "x", y = "y", meta = "col"},
                            x = tsneY[:,1], y = tsneY[:,2], col = gamma_cats),
                            ),
                        )
    # pgfsave("results/countries/plt_tsne_$inf_type.tikz", plt_tsne)
end



begin
    mixing = mean(mswitch[end])
    feature_num = 2
    feature_slice = nmixtures*(feature_num-1)+1:nmixtures*feature_num
    _estimated_dists   = map(g -> Gamma(g[1], inv(g[2])), zip(mean.(mas[end][feature_slice]), mean.(mbs[end][feature_slice])))
    _estimated_mixture = MixtureModel(_estimated_dists, mixing)


    h = fit(Histogram, rand(_estimated_mixture, 10000), nbins=100, closed = :left)
    h = normalize(h, mode=:pdf)

    df = CSV.File("data/Country-data.csv") |> DataFrame
    df = df[!, Not([ :country, :inflation ])]
    fnames = names(df)
    x = range(0, maximum(dataset[:, feature_num]); length = 100)

    plt_mixture = @pgf Axis(
        {
            title="Histogram of $(fnames[feature_num]) ($inf_type)",
            yticklabel_style={
            "/pgf/number format/fixed,
            /pgf/number format/precision=3"
            },
            yminorgrids=true,
            tick_align="outside",
            xtick=range(0, maximum(dataset[:, feature_num]); length=5),
            xmin =0.0,
            xmax=maximum(dataset[:, feature_num]),
            scaled_y_ticks = false,
        },
        Plot({"ybar interval", fill="blue!15", "forget plot", opacity=0.6, "bar width"=0.1}, Table(h)),
        Plot({style="{ultra thick, dashed}", color="magenta"}, Table(x, pdf.(_estimated_mixture, x))), LegendEntry("Mixture"),
        Iterators.flatten([
            [Plot({style="{ultra thick}", color=col}, Table(x, mixing[i]*pdf.(_estimated_dists[i], x))), LegendEntry("$i component")] for (col, i) in zip(colorschemes[:seaborn_bright], 1:nmixtures)
    ])...)
    pgfsave("results/countries/plt_mixture_$inf_type.tikz", plt_mixture)
end