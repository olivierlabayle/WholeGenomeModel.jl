module TestWholeGenomeDataset

using WholeGenomeModel
using Test
using MLUtils
using CSV
using DataFrames
using Random

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

function write_phenotypes(;n=3202)
    tmpdir = mktempdir()
    filepath = joinpath(tmpdir, "phenotypes.csv")
    phenotypes = DataFrame(Y_1 = rand(n))
    CSV.write(filepath, phenotypes)
    return filepath
end
@testset "Test get_dataloaders" begin
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    phenotypes_path = write_phenotypes(;n=3202)
    phenotypes = WholeGenomeModel.phenotypes_from_file(phenotypes_path)
    # Testing DataLoader
    train_loader, val_loader = WholeGenomeModel.get_dataloaders(genotypes_prefix, phenotypes_path;
        phenotypes_id=1,
        variants_batchsize=nothing,
        obs_batchsize=8, 
        splits_ratios=0.7, 
        shuffle_before_split=true,
        shuffle_before_iterate=false,
        rng=Random.default_rng()
    )
    ## Train Loader
    expected_obs = 2241
    @test length(train_loader.data.y) == expected_obs
    @test numobs(train_loader.data.X) == expected_obs
    batch = first(train_loader)
    @test size(batch.y, 1) == 8
    @test size(batch.X, 2) == 8
    first_batch_indices = train_loader.data.X.indices[1:8]
    @test batch.y == phenotypes[first_batch_indices]
    ## Validation Loader
    expected_obs = 961
    @test length(val_loader.data.y) == expected_obs
    @test numobs(val_loader.data.X) == expected_obs
    batch = first(val_loader)
    @test size(batch.y, 1) == 8
    @test size(batch.X, 2) == 8
    first_batch_indices = val_loader.data.X.indices[1:8]
    @test batch.y == phenotypes[first_batch_indices]
end

@testset "Test WholeGenomeDataset" begin
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    snp_datas = WholeGenomeModel.snp_datas_from_prefix(genotypes_prefix)
    variants_batchsize = nothing
    nobs = first(snp_datas).people
    # Test when indices is the whole set of indices
    indices = collect(1:nobs)
    wgd = WholeGenomeDataset(snp_datas, variants_batchsize, indices)
    n_variants = 450
    @test MLUtils.numobs(wgd) == nobs
    ## Test MLUtils.getobs for a single idx 
    idx = 10
    ## The method is only implemented for Vector or UnitRange
    obs_10 = MLUtils.getobs(wgd, [idx])
    ## The expected output is a matrix of shape (n_snps, batchsize)
    @test obs_10 isa Matrix{Float32}
    @test size(obs_10) == (n_variants, 1)
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
    expected_obs_10_15_3 = hcat(obs_10, obs_15, obs_3)
    @test expected_obs_10_15_3 == obs

    # Test with a subset of indices
    indices = [10, 15, 3]
    wgd = WholeGenomeDataset(snp_datas, nothing, indices)
    @test numobs(wgd) == 3
    @test getobs(wgd, [1, 2, 3]) == expected_obs_10_15_3
end

end

true