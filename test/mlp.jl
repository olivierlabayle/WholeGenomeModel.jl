module TestMLP

using Test
using WholeGenomeModel
using MLUtils
using Random
using Optimisers

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

function make_regression(plink_prefix; rng=Random.default_rng(), n_causal_variants=20)
    X = WholeGenomeDataset(plink_prefix)
    p = size(X[[1]], 1)
    θ = zeros(Float32, p)
    θ[rand(rng, 1:p, n_causal_variants)] = 2*rand(rng, Float32, n_causal_variants)
    y = randn(rng, Float32, numobs(X))
    for i in 1:numobs(X)
        y[i] += θ'X[[i]][:, 1]
    end
    return X, permutedims(y)
end

@testset "Test MLP" begin
    plink_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    X, y = make_regression(plink_prefix)

    model = make_whole_genome_model(X, hidden_size=20, optimiser=Adam(0.0001), max_epochs=2000)
    WholeGenomeModel.fit(model, 1, X, y)
end

end

true