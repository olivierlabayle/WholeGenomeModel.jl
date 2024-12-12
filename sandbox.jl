using WholeGenomeModel
using Lux
using Optimisers
using CSV
using DataFrames

data = (
    genotypes_prefix = "/home/s2042526/UK-BioBank-91924/genotyped/ukb22418_c",
    phenotypes_path = "/home/s2042526/UK-BioBank-91924/phenotypes/ir.csv",
    phenotypes_id = "21001-1.0",
    device = cpu_device(),
    variants_batchsize=nothing,
    parallel = true,
)

y = WholeGenomeModel.phenotypes_from_file(data.phenotypes_path; phenotypes_id=data.phenotypes_id)

learner = WholeGenomeModel.learner_from_data(data; hidden_size=20, optimiser=Adam(1e-4))
WholeGenomeModel.fit(learner, data, verbosity=1)

phenotypes = CSV.read(data.phenotypes_path, DataFrame)

missing_idx = findall(ismissing, phenotypes[!, "48-0.0"])
eid_to_idx = Dict(zip(phenotypes.eid, 1:nobs))

snp_datas = WholeGenomeModel.snp_datas_from_prefix(data.genotypes_prefix)

bed_person_info = DataFrame(
    iid = string.(snp_datas[1].person_info.iid),
    bed_person_idx = 1:nrow(snp_datas[1].person_info)
)
phenotypes_person_info = DataFrame(
    eid = string.(phenotypes.eid),
    phenotypes_person_idx = 1:nrow(phenotypes)
)
joined_person_indo = innerjoin(
    bed_person_info, 
    phenotypes_person_info, 
    on=:iid =>:eid
)