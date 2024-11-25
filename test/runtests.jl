using Test
using WholeGenomeModel

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")
@testset "Tests WholeGenomeModel" begin
    @test include(joinpath(TESTDIR, "whole_genome_dataset.jl"))
end