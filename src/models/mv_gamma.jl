export MvGammaMixtureModelParameters, mv_gamma_mixture_model, mv_gamma_mixture_inference

using GraphPPL
using ReactiveMP
using Rocket
using Distributions
using Parameters

import ProgressMeter

@with_kw struct MvGammaMixtureModelParameters
    # required
    nmixtures  
    nfeatures  
    priors_as    
    priors_bs 


    # optional 
    T                = Float64
    meta             = GaussLaguerreQuadrature(Float64, 128)
    as_prod_strategy = FoldLeftProdStrategy()
    bs_prod_strategy = FoldLeftProdStrategy()
    as_constraint    = Marginalisation()
    bs_constraint    = Marginalisation()
end

@model [ default_factorisation = MeanField() ] function mv_gamma_mixture_model(nobservations, parameters::MvGammaMixtureModelParameters)

    nmixtures = parameters.nmixtures
    nfeatures = parameters.nfeatures
    priors_as = parameters.priors_as
    priors_bs = parameters.priors_bs
    
    T                = parameters.T
    meta             = parameters.meta
    as_prod_strategy = parameters.as_prod_strategy
    bs_prod_strategy = parameters.bs_prod_strategy
    as_constraint    = parameters.as_constraint
    bs_constraint    = parameters.bs_constraint

    # specify dirichlet prior on cluster assignment
    s ~ Dirichlet(fill(one(T), nmixtures))

    # specify random variables for mixtures (use remainder and modulo to fetch later on)
    as = randomvar(nmixtures * nfeatures, prod_strategy = as_prod_strategy, constraint = as_constraint)
    bs = randomvar(nmixtures * nfeatures, prod_strategy = bs_prod_strategy, constraint = bs_constraint)

    for i in 1:nmixtures * nfeatures
        as[i] ~ GammaShapeRate(convert(T, shape(priors_as[i])), convert(T, rate(priors_as[i]))) 
        bs[i] ~ GammaShapeRate(convert(T, shape(priors_bs[i])), convert(T, rate(priors_bs[i])))
    end

    # specify selector variables (one for each feature vector)
    z = randomvar(nobservations)

    # specify observations as data variables
    y = datavar(T, nobservations * nfeatures)

    # specify metadata
    # rng  = MersenneTwister(approximation_points)
    # meta = GammaMixtureNodeMetadata(ImportanceSamplingApproximation(rng, approximation_points))
    meta = GammaMixtureNodeMetadata(meta)

    # prepare as and bs
    tas = [tuple(as[((k-1)*nmixtures+1):(k*nmixtures)]...) for k=1:nfeatures]
    tbs = [tuple(bs[((k-1)*nmixtures+1):(k*nmixtures)]...) for k=1:nfeatures]

    # define selector variables
    for i in 1:nobservations
        z[i] ~ Categorical(s)
    end

    # define likelihood over observations
    for i in 1:nobservations*nfeatures
        id_observation = (i-1)÷nfeatures + 1
        id_feature = rem(i-1, nfeatures) + 1
        y[i] ~ GammaMixture(z[id_observation], tas[id_feature], tbs[id_feature]) where { meta = meta }
    end

    # specify scheduler
    scheduler = schedule_updates(as, bs)

    # return variables
    return scheduler, s, as, bs, z, y
end

function mv_gamma_mixture_inference(data, niterations, parameters)

    @unpack nmixtures, T = parameters
    @unpack priors_as, priors_bs = parameters

    # fetch number of observations and number of features
    (nobservations, nfeatures) = size(data)

    # create model
    model, (scheduler, s, as, bs, z, y) = mv_gamma_mixture_model(nobservations, parameters, options=(limit_stack_depth=100, ));

    # allocate marginals
    mswitch   = keep(Marginal)
    mselector = keep(Vector{Marginal})
    mas       = keep(Vector{Marginal})
    mbs       = keep(Vector{Marginal})
    fe        = ScoreActor(T)

    fe_scheduler = PendingScheduler()

    # subscribe to marginals
    switch_sub    = subscribe!(getmarginal(s), mswitch)
    selectors_sub = subscribe!(getmarginals(z), mselector)
    as_sub        = subscribe!(getmarginals(as), mas)
    bs_sub        = subscribe!(getmarginals(bs), mbs)
    fe_sub        = subscribe!(score(T, BetheFreeEnergy(), model, fe_scheduler), fe)

    # set initial marginals
    setmarginal!(s, convert(Dirichlet{T}, vague(Dirichlet, nmixtures)))

    @assert length(as) === length(bs)
    @assert length(as) === length(priors_as)
    @assert length(priors_as) === length(priors_bs)

    for i in 1:nmixtures * nfeatures
        setmarginal!(as[i], infgamma(T, 1.0, ϵ = 1.0))
        setmarginal!(bs[i], infgamma(T, 1.0, ϵ = 1.0))
    end

    # perform reactive message pasing (first flatten data)
    tdata = map(x -> T(x), [data'...])

    ProgressMeter.@showprogress for i in 1:niterations
        update!(y, tdata)
        release!(scheduler)
        release!(fe_scheduler)
    end

    # unsubscribe from subscriptions
    unsubscribe!(fe_sub)
    unsubscribe!(switch_sub)
    unsubscribe!(selectors_sub)
    unsubscribe!(as_sub)
    unsubscribe!(bs_sub)

    # return obtained values
    return map(getvalues, (mswitch, mselector, mas, mbs, fe))
end