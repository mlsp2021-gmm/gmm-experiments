export GammaMixtureModelParameters, gamma_mixture_model, gamma_mixture_inference

using GraphPPL
using ReactiveMP
using Rocket
using Distributions
using Parameters

import ProgressMeter

@with_kw struct GammaMixtureModelParameters
    # required
    nmixtures  
    priors_as    
    priors_bs 
    prior_s

    # optional 
    T                = Float64
    meta             = GaussLaguerreQuadrature(Float64, 128)
    as_prod_strategy = FoldLeftProdStrategy()
    bs_prod_strategy = FoldLeftProdStrategy()
    as_constraint    = Marginalisation()
    bs_constraint    = Marginalisation()
end


@model [ default_factorisation = MeanField() ] function gamma_mixture_model(nobservations, parameters::GammaMixtureModelParameters)

    nmixtures = parameters.nmixtures
    priors_as = parameters.priors_as
    priors_bs = parameters.priors_bs
    prior_s   = parameters.prior_s
    
    T                = parameters.T
    meta             = parameters.meta
    as_prod_strategy = parameters.as_prod_strategy
    bs_prod_strategy = parameters.bs_prod_strategy
    as_constraint    = parameters.as_constraint
    bs_constraint    = parameters.bs_constraint

    s ~ Dirichlet(convert(AbstractArray{T}, probvec(prior_s)))

    as = randomvar(nmixtures, prod_strategy = as_prod_strategy, constraint = as_constraint)
    bs = randomvar(nmixtures, prod_strategy = bs_prod_strategy, constraint = bs_constraint)

    for i in 1:nmixtures
        as[i] ~ GammaShapeRate(convert(T, shape(priors_as[i])), convert(T, rate(priors_as[i])))
        bs[i] ~ GammaShapeRate(convert(T, shape(priors_bs[i])), convert(T, rate(priors_bs[i])))
    end

    z = randomvar(nobservations)
    y = datavar(T, nobservations)

    meta = GammaMixtureNodeMetadata(meta)

    tas = tuple(as...)
    tbs = tuple(bs...)

    for i in 1:nobservations
        z[i] ~ Categorical(s)
        y[i] ~ GammaMixture(z[i], tas, tbs) where { meta = meta }
    end

    scheduler = schedule_updates(z, s, bs, as) # May influence results: TODO

    return scheduler, s, as, bs, z, y
end

function gamma_mixture_inference(data, niterations, parameters; with_progress = true)

    @unpack nmixtures, T = parameters
    @unpack priors_as, priors_bs, prior_s = parameters

    # fetch number of observations and number of features
    nobservations = length(data)

    # create model
    model, (scheduler, s, as, bs, z, y) = gamma_mixture_model(nobservations, parameters, options=(limit_stack_depth=100, ));

    # allocate marginals
    mswitch   = keep(Marginal)
    mselector = buffer(Marginal, nobservations)
    mas       = buffer(Marginal, nmixtures)
    mbs       = buffer(Marginal, nmixtures)
    fe        = ScoreActor(T)

    fe_scheduler = PendingScheduler()

    # subscribe to marginals
    switch_sub    = subscribe!(getmarginal(s), mswitch)
    selectors_sub = subscribe!(getmarginals(z), mselector)
    as_sub        = subscribe!(getmarginals(as), mas)
    bs_sub        = subscribe!(getmarginals(bs), mbs)
    fe_sub        = subscribe!(score(T, BetheFreeEnergy(), model, fe_scheduler), fe)

    setmarginal!(s, prior_s)
    setmarginals!(z, convert(Categorical{T, Vector{T}}, vague(Categorical, nmixtures)))

    for (a, b, pa, pb) in zip(as, bs, priors_as, priors_bs)
        setmarginal!(b, GammaShapeRate(1.0, 1.0))
    end

    tdata = map(x -> T(x), data)

    progress = ProgressMeter.Progress(niterations, 1)

    for i in 1:niterations
        update!(y, tdata)
        release!(scheduler)
        release!(fe_scheduler)
        if with_progress
            ProgressMeter.next!(progress)
        end
    end

    # unsubscribe from subscriptions
    # unsubscribe!(fe_sub)
    # unsubscribe!(switch_sub)
    # unsubscribe!(selectors_sub)
    # unsubscribe!(as_sub)
    # unsubscribe!(bs_sub)

    # return obtained values

    # hack for backward compatibility with old scripts
    output = map(getvalues, (mswitch, mselector, mas, mbs, fe))
    return (output[1], [ output[2] ], [ output[3] ], [ output[4] ], output[5])
end