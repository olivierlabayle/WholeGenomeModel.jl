struct WholeGenomeDataset{T <: Union{Nothing, Int}}
    snp_datas::Vector{SnpData}
    snp_batchsize::T
    indices::Vector{Int}
end

MLUtils.numobs(wgd::WholeGenomeDataset) = length(wgd.indices)

get_whole_snps_matrix(wgd::WholeGenomeDataset, idx) = mapreduce(
    snp_data -> convert(Matrix{Float32}, @view(snp_data.snparray[wgd.indices[idx], :])),
    hcat,
    wgd.snp_datas
)

Base.getindex(wgd::WholeGenomeDataset{Nothing}, idx::Union{Vector{Int}, UnitRange{Int}}) =
    permutedims(get_whole_snps_matrix(wgd, idx))

Base.getindex(wgd::WholeGenomeDataset{Int}, idx) = 
    throw(error("Not implemented yet."))

function snp_datas_from_prefix(genotypes_prefix)
    genotypes_dir, file_prefix = splitdir(genotypes_prefix)
    bed_files = filter(
            f -> startswith(f, file_prefix) && endswith(f, "bed"), 
            readdir(genotypes_dir)
        )
    return [SnpData(bed_file[1:end-4]) 
        for bed_file in joinpath.(genotypes_dir, bed_files)]
end

function phenotypes_from_file(phenotypes_path; phenotypes_id=1)
    if endswith(phenotypes_path, "csv")
        return CSV.read(phenotypes_path, DataFrame; select=[phenotypes_id])[!, 1]
    else
        throw(ArgumentError("Only (.csv, ) files are supported."))
    end
end

function get_dataloaders(genotypes_prefix, phenotypes_path;
    phenotypes_id=1,
    variants_batchsize=nothing,
    obs_batchsize=8, 
    splits_ratios=0.7, 
    shuffle_before_split=true,
    shuffle_before_iterate=true,
    rng=Random.default_rng()
    )
    snp_datas = WholeGenomeModel.snp_datas_from_prefix(genotypes_prefix)
    phenotypes = WholeGenomeModel.phenotypes_from_file(phenotypes_path; phenotypes_id=phenotypes_id)
    nobs = size(phenotypes, 1)
    splits_indices = splitobs(1:nobs; at=splits_ratios, shuffle=shuffle_before_split)
    return map(splits_indices) do split_indices
        split_indices = collect(split_indices)
        println(length(split_indices))
        X = WholeGenomeDataset(snp_datas, variants_batchsize, split_indices)
        y = phenotypes[split_indices]
        DataLoader((X=X, y=y), batchsize=obs_batchsize, rng=rng, shuffle=shuffle_before_iterate)
    end
end