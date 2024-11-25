module TestWholeGenomeDataset

using WholeGenomeModel
using Test
using MLUtils

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

@testset "Test WholeGenomeDataset" begin
    plink_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    wgd = WholeGenomeDataset(plink_prefix)
    expected_n = 3202
    expected_n_snps = 450
    @test MLUtils.numobs(wgd) == expected_n

    # Test MLUtils.getobs for a single idx 
    idx = 10
    ## The method is only implemented for Vector or UnitRange
    obs_10 = MLUtils.getobs(wgd, [idx])
    ## The expected output is a matrix of shape (n_snps, batchsize)
    @test obs_10 isa Matrix{Float32}
    @test size(obs_10) == (expected_n_snps, 1)
    ## The expected output is a concatenation of all snps from all files
    expected_obs_10_snps_1_to_100 = convert(Vector{Float32}, view(wgd.snp_datas[1].snparray, idx, :))
    expected_obs_10_snps_101_to_250 = convert(Vector{Float32}, view(wgd.snp_datas[2].snparray, idx, :))
    expected_obs_10_snps_251_to_450 = convert(Vector{Float32}, view(wgd.snp_datas[3].snparray, idx, :))
    expected_obs_10 = reshape(vcat(expected_obs_10_snps_1_to_100, expected_obs_10_snps_101_to_250, expected_obs_10_snps_251_to_450), :, 1)
    @test obs_10 == expected_obs_10
    ## Check observations are stacked in correct order
    obs = MLUtils.getobs(wgd, [10, 15, 3])
    obs_15 = MLUtils.getobs(wgd, [15])
    obs_3 = MLUtils.getobs(wgd, [3])
    expected_obs = hcat(obs_10, obs_15, obs_3)
    @test expected_obs == obs

    # Testing DataLoader
    batchsize = 10
    y = rand(Float32, MLUtils.numobs(wgd))
    wgdl = DataLoader((genotypes=wgd, y=y); batchsize=batchsize, shuffle=true, parallel=false)
    iterated_obs = collect(wgdl)
    @test size(iterated_obs, 1) == ceil(3202 / batchsize)
end

end

true