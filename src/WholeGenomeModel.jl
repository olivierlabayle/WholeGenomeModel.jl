module WholeGenomeModel

using Lux
using Zygote
using Optimisers
using Printf
using MLUtils
using SnpArrays

include("whole_genome_dataset.jl")

# MLP = @compact(
#     block_chain = Lux.Chain(Lux.Dense(50, 10, tanh), Lux.Dense(10, 1)),
#     out_chain = Lux.Dense(4, 1)
#     ) do blocks
#     yblocks = mapreduce(block_chain, vcat, blocks)
#     out = out_chain(yblocks)
#     @return out
# end
             
export MLP
export WholeGenomeDataset

end
