module WholeGenomeModel

using Lux
using Zygote
using Optimisers
using Printf
using MLUtils
using SnpArrays
using Random
using CSV
using DataFrames

include("whole_genome_dataset.jl")
include("mlp.jl")
             
export MLP
export WholeGenomeDataset
export DLWholeGenomeRegression
export make_whole_genome_model

end
