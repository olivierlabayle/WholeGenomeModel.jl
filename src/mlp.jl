
struct DLWholeGenomeRegression 
    model
    optimiser
    patience::Int
    batchsize::Int
    max_epochs::Int
    train_ratio::Float64
    shuffle_before_split::Bool
    shuffle_before_iterate::Bool
    rng
end

DLWholeGenomeRegression(model; 
    optimiser=Adam(),
    patience=10,
    batchsize=8, 
    max_epochs=1000, 
    train_ratio=0.8,
    shuffle_before_split=true,
    shuffle_before_iterate=true,
    rng=Random.default_rng()
    ) = DLWholeGenomeRegression(model, 
    optimiser,
    patience,
    batchsize,
    max_epochs,
    train_ratio,
    shuffle_before_split,
    shuffle_before_iterate,
    rng
    )

function learner_from_data(data::NamedTuple, model_type=MLP; 
    optimiser=Adam(),
    patience=10,
    batchsize=8, 
    max_epochs=1000, 
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
        patience=patience,
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
        parallel=data.parallel,
        rng=learner.rng
    )
    loss_fn = MSELoss()
    best = (epoch=0, val_loss=Inf, parameters=ps)
    train_losses = Float32[]
    val_losses = Float32[]
    for epoch in 1:learner.max_epochs
        # Train on training set
        epoch_train_loss = 0.
        for (x, y) in data.device(train_loader)
            gs, train_loss, stats, train_state = Training.single_train_step!(
                AutoZygote(), 
                loss_fn,
                (x, y), 
                train_state
            )
            epoch_train_loss += train_loss
        end
        push!(train_losses, epoch_train_loss)

        # Evaluate on validation set
        st_ = Lux.testmode(train_state.states)
        epoch_val_loss = 0.
        for (x, y) in val_loader
            ŷ, st_ = learner.model(x, train_state.parameters, st_)
            epoch_val_loss += loss_fn(ŷ, y)
        end
        # Update Loss Vectors
        push!(val_losses, epoch_val_loss)
        # Log
        verbosity > 1 && @info(string("Training Loss after epoch ", epoch, ": ", epoch_train_loss))
        verbosity > 0 && @info(string("Validation Loss after epoch ", epoch, ": ", epoch_val_loss))
        # Update best, stop if patience is reached or continue
        if epoch_val_loss < best.val_loss
            best = (epoch=epoch, val_loss=epoch_val_loss, parameters=deepcopy(ps))
        elseif (epoch - best.epoch) > learner.patience
            verbosity > 0 && @info("Patience reached, stopping.")
            return best
        end
    end
    return best
end

function MLP(input_size; hidden_size=10)
    return Chain(
        Dense(input_size, hidden_size, tanh), 
        Dense(hidden_size, 1)
    )
end

