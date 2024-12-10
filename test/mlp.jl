module TestMLP

using Test
using WholeGenomeModel
using MLUtils
using Random
using Optimisers
using CSV
using DataFrames
using Lux

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

function make_regression(genotypes_prefix; rng=Random.default_rng(), n_causal_variants=20)
    X = WholeGenomeDataset(genotypes_prefix)
    p = size(X[[1]], 1)
    θ = zeros(Float32, p)
    θ[rand(rng, 1:p, n_causal_variants)] = 2*rand(rng, Float32, n_causal_variants)
    y = randn(rng, Float32, numobs(X))
    for i in 1:numobs(X)
        y[i] += θ'X[[i]][:, 1]
    end
    tmpdir = mktempdir()
    phenotypes_path = joinpath(tmpdir, "phenotypes.csv")
    CSV.write(phenotypes_path, DataFrame(Y=y))
    return phenotypes_path
end

@testset "Test MLP" begin
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    phenotypes_path = make_regression(genotypes_prefix)
    
    data = (
        genotypes_prefix = genotypes_prefix,
        phenotypes_path = phenotypes_path,
        phenotypes_id = 1,
        device = cpu_device(),
        parallel = true,
        variants_batchsize = nothing
    )

    learner = WholeGenomeModel.learner_from_data(data; hidden_size=20, optimiser=Adam(1e-4))
    WholeGenomeModel.fit(learner, data, verbosity=1)
end

end

true