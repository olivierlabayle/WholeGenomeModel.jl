
struct DLWholeGenomeRegression 
    model
    optimiser
    batchsize::Int
    max_epochs::Int
    train_ratio::Float64
    shuffle_before_split::Bool
    shuffle_before_iterate::Bool
    rng
end

DLWholeGenomeRegression(model; 
    optimiser=Adam(),
    batchsize=8, 
    max_epochs=1000, 
    train_ratio=0.8,
    shuffle_before_split=true,
    shuffle_before_iterate=true,
    rng=Random.default_rng()
    ) = DLWholeGenomeRegression(model, 
    optimiser, 
    batchsize, 
    max_epochs, 
    train_ratio,
    shuffle_before_split,
    shuffle_before_iterate,
    rng
    )

function learner_from_data(data::NamedTuple, model_type=MLP; 
    optimiser=Adam(),
    batchsize=8, 
    max_epochs=10^6, 
    train_ratio=0.8,
    shuffle_before_split=true,
    shuffle_before_iterate=true,
    rng=Random.default_rng(),
    model_kwargs...
    )
    # y = phenotypes_from_file(data.phenotypes_path, phenotypes_id=data.phenotypes_id)
    X = WholeGenomeDataset(data.genotypes_prefix; 
        variants_batchsize=data.variants_batchsize,
        indices=nothing
    )
    input_size = size(X[[1]], 1)
    model = model_type(input_size; model_kwargs...)
    return DLWholeGenomeRegression(model;
        optimiser=optimiser, 
        batchsize=batchsize, 
        max_epochs=max_epochs, 
        train_ratio=train_ratio,
        shuffle_before_split=shuffle_before_split,
        shuffle_before_iterate=shuffle_before_iterate,
        rng=rng
    )
end

function fit(learner::DLWholeGenomeRegression, data; verbosity=0)
    ps, st = Lux.setup(learner.rng, learner.model) |> data.device
    train_state = Lux.Training.TrainState(learner.model, ps, st, learner.optimiser)
    train_loader, val_loader = get_dataloaders(data.genotypes_prefix, data.phenotypes_path;
        phenotypes_id=data.phenotypes_id,
        variants_batchsize=data.variants_batchsize,
        obs_batchsize=learner.batchsize, 
        splits_ratios=learner.train_ratio, 
        shuffle_before_split=learner.shuffle_before_split,
        shuffle_before_iterate=learner.shuffle_before_iterate,
        rng=learner.rng
    )
    for epoch in 1:learner.max_epochs
        epoch_loss = 0.
        for (x, y) in data.device(train_loader)
            gs, loss, stats, train_state = Training.single_train_step!(
                AutoZygote(), 
                MSELoss(),
                (x, y), 
                train_state
            )
            epoch_loss += loss
        end
        verbosity > 0 && @info(string("Training Loss after epoch ", epoch, ": ", epoch_loss))
    end
    return ps
end

function MLP(input_size; hidden_size=10)
    return Chain(
        Dense(input_size, hidden_size, tanh), 
        Dense(hidden_size, 1)
    )
end

