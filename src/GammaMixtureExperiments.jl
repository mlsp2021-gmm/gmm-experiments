module GammaMixtureExperiments

using ReactiveMP

## Hotfix for ReactiveMP and savename from DrWatson
Base.string(meta::ImportanceSamplingApproximation) = "sampling($(meta.nsamples))"
Base.string(::EM)              = "em"
Base.string(::Marginalisation) = "vi"

include("utils.jl")
include("datasets.jl")

include("models/gamma.jl")
include("models/mv_gamma.jl")

end