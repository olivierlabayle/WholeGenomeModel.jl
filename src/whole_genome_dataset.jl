struct WholeGenomeDataset{T <: Union{Nothing, Int}}
    snp_datas::Vector{SnpData}
    nobs::Int
    snp_batchsize::T
    function WholeGenomeDataset(plink_prefix, snp_batchsize=nothing)
        genotypes_dir, file_prefix = splitdir(plink_prefix)
        bed_files = filter(
            f -> startswith(f, file_prefix) && endswith(f, "bed"), 
            readdir(genotypes_dir)
        )
        snp_datas = [SnpData(bed_file[1:end-4]) 
            for bed_file in joinpath.(genotypes_dir, bed_files)]
        nobs = first(snp_datas).people
        return new{typeof(snp_batchsize)}(snp_datas, nobs, snp_batchsize)
    end
end

MLUtils.numobs(wgd::WholeGenomeDataset) = wgd.nobs

get_whole_snps_matrix(wgd::WholeGenomeDataset, idx) = mapreduce(
    snp_data -> convert(Matrix{Float32}, @view(snp_data.snparray[idx, :])),
    hcat,
    wgd.snp_datas
)

Base.getindex(wgd::WholeGenomeDataset{Nothing}, idx::Union{Vector{Int}, UnitRange{Int}}) =
    permutedims(get_whole_snps_matrix(wgd, idx))

Base.getindex(wgd::WholeGenomeDataset{Int}, idx) = 
    throw(error("Not implemented yet."))