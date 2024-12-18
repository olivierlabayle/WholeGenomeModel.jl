module TestMLP

using Test
using WholeGenomeModel
using MLUtils
using Random
using Optimisers
using CSV
using DataFrames
using Lux
using TOML

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test MLP" begin
    # Setup data and config
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    phenotypes_path = write_linear_phenotypes(genotypes_prefix)
    config = Dict(
        "global" => Dict(
            "rng" => 123,
            "verbosity" => 2
        ),
        "training" => Dict(
            "train_ratio" => 0.8,
            "patience" => 10,
            "batchsize" => 8,
            "max_epochs" => 1000,
            "shuffle_before_iterate" => true,
            "parallel" => true,
            "device" => "cpu"  
        ),
        "data" => Dict(
            "genotypes_prefix" => joinpath(TESTDIR, "assets", "unphased_bed", "ukb_"),
            "phenotypes_path" => phenotypes_path,
            "phenotypes_id" => "Y",
            "shuffle_before_split" => true,
        ),
        "optimiser" => Dict(
            "name" => "Adam",
            "eta"  => 5e-4
        ),
        "model" => Dict(
            "name" => "MLP",
            "hidden_size" => 10
        )
    )
    tmpdir = mktempdir()
    configfile = joinpath(tmpdir, "config.toml")
    open(io -> TOML.print(io, config), configfile, "w")
    main(configfile)
end

end

true