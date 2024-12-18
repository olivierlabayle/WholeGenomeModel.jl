module WholeGenomeModel

using Lux
using Zygote
using Optimisers
using Printf
using MLUtils
using SnpArrays
using Random
using CSV
using DataFrames
using TOML

include("whole_genome_dataset.jl")
include("mlp.jl")
             
export MLP
export GenotypingDataset, make_datasets
export fit, make_learner, main

function (@main)(ARGS)
    configfile = ARGS[1]
    config = TOML.parsefile(configfile)
    global_config = pop!(config, "global")
    optim_config = pop!(config, "optimiser")
    model_config = pop!(config, "model")
    train_config = pop!(config, "training")
    data_config = pop!(config, "data")
    verbosity = haskey(global_config, "verbosity") ? pop!(global_config, "verbosity") : 1
    # Set Random seed
    rng = haskey(global_config, "rng") ? pop!(global_config, "rng") : 123
    Random.seed!(rng)
    # Make datasets
    verbosity > 0 && @info "Making datasets."
    genotypes_prefix = pop!(data_config, "genotypes_prefix")
    phenotypes_path = pop!(data_config, "phenotypes_path")
    data_config = NamedTuple((Symbol(key),value) for (key, value) in data_config)
    train_dataset, val_dataset = make_datasets(
        genotypes_prefix, 
        phenotypes_path;
        data_config...
        )
    # Make Learner
    verbosity > 0 && @info "Making Learner."
    ## Optimisation config
    optimiser_type = eval(Symbol(pop!(optim_config, "name")))
    optim_config = NamedTuple((Symbol(key),value) for (key, value) in optim_config)
    optimiser = optimiser_type(;optim_config...)
    ## General learning config
    train_config["device"] = train_config["device"] == "cpu" ? cpu_device() : gpu_device()
    train_config = NamedTuple((Symbol(key),value) for (key, value) in train_config)
    ## Model Config
    model_type = eval(Symbol(pop!(model_config, "name")))
    model_config = NamedTuple((Symbol(key),value) for (key, value) in model_config)
    learner = make_learner(train_dataset; 
        model_type=model_type,
        optimiser=optimiser,
        model_config...,
        train_config...
    )
    # Fit Learner
    verbosity > 0 && @info "Fitting Learner."
    dataset = (train_dataset=train_dataset, val_dataset=val_dataset)
    fit(learner, dataset; verbosity=verbosity)
    verbosity > 0 && @info "Done."
end

end
