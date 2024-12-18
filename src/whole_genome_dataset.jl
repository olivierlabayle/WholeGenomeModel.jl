struct GenotypingDataset{B <: Union{Nothing, Int}, I <: Union{Nothing, Vector{Int}}}
    snp_datas::Vector{SnpData}
    variants_batchsize::B
    indices::I
end

function GenotypingDataset(genotypes_prefix; variants_batchsize=nothing, indices=nothing)
    snp_datas = snp_datas_from_prefix(genotypes_prefix)
    return GenotypingDataset(snp_datas, variants_batchsize, indices)
end

MLUtils.numobs(wgd::GenotypingDataset) = length(wgd.indices)

MLUtils.numobs(wgd::GenotypingDataset{B, Nothing}) where {B <: Union{Nothing, Int}} = first(wgd.snp_datas).people

map_indices(dataset_indices::Nothing,  indices) = indices

map_indices(dataset_indices, indices) = dataset_indices[indices]

get_whole_snps_matrix(wgd::GenotypingDataset, idx) = mapreduce(
    snp_data -> convert(Matrix{Float32}, @view(snp_data.snparray[map_indices(wgd.indices, idx), :]), impute=true),
    hcat,
    wgd.snp_datas
)

Base.getindex(wgd::GenotypingDataset{Nothing}, idx::Union{Vector{Int}, UnitRange{Int}}) =
    permutedims(get_whole_snps_matrix(wgd, idx))

Base.getindex(wgd::GenotypingDataset{Int}, idx) = 
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

function read_phenotypes(phenotypes_path; phenotypes_id=:Y)
    if endswith(phenotypes_path, ".csv")
        return CSV.read(
            phenotypes_path, 
            DataFrame; 
            types=Dict(:FID => String, :IID => String), 
            select=string.(["FID", "IID", phenotypes_id])
        )
    else
        throw(ArgumentError("Only (.csv, ) files are supported."))
    end
end

function make_datasets(genotypes_prefix, phenotypes_path;
    phenotypes_id=:Y,
    variants_batchsize=nothing,
    splits_ratios=0.7, 
    shuffle_before_split=true,
    )
    # Load data
    snp_datas = WholeGenomeModel.snp_datas_from_prefix(genotypes_prefix)
    phenotypes_dataset = WholeGenomeModel.read_phenotypes(phenotypes_path; phenotypes_id=phenotypes_id)
    # Map phenotypes to genotypes
    mapped_phenotypes_dataset = select(
        first(snp_datas).person_info, 
        :fid =>:FID, 
        :iid => :IID,
    )
    mapped_phenotypes_dataset.ID = 1:nrow(mapped_phenotypes_dataset)
    leftjoin!(
        mapped_phenotypes_dataset, 
        phenotypes_dataset, 
        on=[:FID, :IID]
    )
    # Drop missing phenotypes
    dropmissing!(mapped_phenotypes_dataset)
    # Build dataloaders
    splits_indices = splitobs(mapped_phenotypes_dataset.ID; at=splits_ratios, shuffle=shuffle_before_split)
    return map(splits_indices) do split_indices
        split_indices = collect(split_indices)
        (
            X = GenotypingDataset(snp_datas, variants_batchsize, split_indices),
            y = permutedims(Float32.(mapped_phenotypes_dataset[split_indices, end]))
        )
    end
end