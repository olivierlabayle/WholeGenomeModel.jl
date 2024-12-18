module TestGenotypingDataset

using WholeGenomeModel
using Test
using MLUtils
using CSV
using DataFrames
using Random

TESTDIR = joinpath(pkgdir(WholeGenomeModel), "test")

include(joinpath(TESTDIR, "testutils.jl"))
@testset "Test GenotypingDataset" begin
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    snp_datas = WholeGenomeModel.snp_datas_from_prefix(genotypes_prefix)
    variants_batchsize = nothing
    nobs = first(snp_datas).people
    # Test when indices is the whole set of indices
    indices = nothing
    wgd = GenotypingDataset(snp_datas, variants_batchsize, indices)
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
    wgd = GenotypingDataset(snp_datas, nothing, indices)
    @test numobs(wgd) == 3
    @test getobs(wgd, [1, 2, 3]) == expected_obs_10_15_3
end

@testset "Test make_datasets" begin
    # Prepare Data
    genotypes_prefix = joinpath(TESTDIR, "assets", "unphased_bed", "ukb_")
    phenotypes_path = write_linear_phenotypes(genotypes_prefix)
    phenotypes = WholeGenomeModel.read_phenotypes(phenotypes_path)
    # Build datasets
    train_dataset, val_dataset = make_datasets(genotypes_prefix, phenotypes_path, shuffle_before_split=false)
    # Test numobs
    ## The joint size of datasets is: 
    ## number of individuals in the genotyping data - number of missing phenotypes - number of non matching FID/IID
    train_size = numobs(train_dataset)
    val_size = numobs(val_dataset)
    joint_size = train_size + val_size
    @test joint_size  == 3202 - 10 - 1 == 3191
    @test 0.69 <= train_size/joint_size <= 0.71
    @test 0.29 <= val_size/joint_size <= 0.31
    # Test getobs
    # IIDs in phenotypes are reversed compared to genotypes, 
    # so we need to join to get the matching data 
    idx = [3, 10, 15]
    x, y = getobs(train_dataset, idx)
    snp_data = first(train_dataset.X.snp_datas).person_info
    expected_y = leftjoin!(
        DataFrame(FID = snp_data[idx, :fid], IID=snp_data[idx, :iid]),
        phenotypes,
        on=[:FID, :IID]
    ).Y
    @test y == permutedims(Float32.(expected_y))
    @test x == getobs(train_dataset.X, idx)
    # The last observation of the validation dataset corresponds to the first non missing phenotype
    idx = [numobs(val_dataset)]
    x, y = getobs(val_dataset, idx)
    @test y == permutedims(Float32.(phenotypes[[12], :Y]))
    @test x == getobs(val_dataset.X, idx)
end

end

true