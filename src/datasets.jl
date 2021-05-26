export CountriesData
export load_dataset

using CSV
using DataFrames

abstract type RealDataset end

struct CountriesData <: RealDataset end

getpath(::CountriesData) = "data/Country-data.csv"

function load_dataset(dataset::RealDataset, exclude; force_abs = false)
    df = CSV.File(getpath(dataset)) |> DataFrame
    df = select(df, exclude)
    data = Matrix(df)
    if force_abs
        data = abs.(data)
    end
    return Float64.(data)[:, 1:end]
end