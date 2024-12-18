
function write_linear_phenotypes(genotypes_prefix; rng=Random.default_rng(), n_causal_variants=20)
    X = GenotypingDataset(genotypes_prefix)
    n = numobs(X)
    person_info = first(X.snp_datas).person_info
    # Make phenotype
    p = size(X[[1]], 1)
    θ = zeros(Float32, p)
    θ[rand(rng, 1:p, n_causal_variants)] = 2*rand(rng, Float32, n_causal_variants)
    y = randn(rng, Float32, numobs(X))
    for i in 1:numobs(X)
        y[i] += θ'X[[i]][:, 1]
    end
    # Add some missingness in phenotype
    y = vcat(repeat([missing], 10), rand(n-10))
    # Mix FID/IID order and add some non-matching FID/IID
    phenotypes = DataFrame(
        FID=reverse(person_info.fid),
        IID=reverse(person_info.iid),
        Y = y
    )
    phenotypes.IID[11] = "ID_NOT_IN_GENOTYPES"
    # Write phenotypes
    tmpdir = mktempdir()
    filepath = joinpath(tmpdir, "phenotypes.csv")
    CSV.write(filepath, phenotypes)
    return filepath
end