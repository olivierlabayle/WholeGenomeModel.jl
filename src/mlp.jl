
struct DLWholeGenomeRegression 
    model
    optimiser
    patience::Int
    batchsize::Int
    max_epochs::Int
    train_ratio::Float64
    shuffle_before_iterate::Bool
    parallel::Bool
    device
end

get_input_size(dataset) = size(dataset.X[[1]], 1)

function make_learner(dataset; 
    model_type=MLP, 
    optimiser=Adam(),
    patience=10,
    batchsize=8, 
    max_epochs,
    train_ratio,
    shuffle_before_iterate=true,
    parallel=true,
    device=cpu_device(),
    model_kwargs...
    )
    return DLWholeGenomeRegression(
        model_type(get_input_size(dataset); model_kwargs...),
        optimiser,
        patience,
        batchsize,
        max_epochs,
        train_ratio,
        shuffle_before_iterate,
        parallel,
        device
    )
end

function fit(learner::DLWholeGenomeRegression, data; verbosity=0)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, learner.model) |> learner.device
    train_state = Lux.Training.TrainState(learner.model, ps, st, learner.optimiser)
    
    loss_fn = MSELoss()

    train_loader = DataLoader(data.train_dataset, 
        batchsize=learner.batchsize, 
        shuffle=learner.shuffle_before_iterate,
        parallel=learner.parallel,
        rng=rng
    )
    val_loader = DataLoader(data.val_dataset, 
        batchsize=learner.batchsize, 
        shuffle=learner.shuffle_before_iterate,
        parallel=learner.parallel,
        rng=rng
    )
    best = (epoch=0, val_loss=Inf, parameters=ps)
    train_losses = Float32[]
    val_losses = Float32[]
    for epoch in 1:learner.max_epochs
        # Train on training set
        epoch_train_loss = 0.
        for (x, y) in learner.device(train_loader)
            gs, train_loss, stats, train_state = Training.single_train_step!(
                AutoZygote(), 
                loss_fn,
                (x, y), 
                train_state
            )
            epoch_train_loss += train_loss
        end
        push!(train_losses, epoch_train_loss/numobs(train_loader))

        # Evaluate on validation set
        st_ = Lux.testmode(train_state.states)
        epoch_val_loss = 0.
        for (x, y) in val_loader
            ŷ, st_ = learner.model(x, train_state.parameters, st_)
            epoch_val_loss += loss_fn(ŷ, y)
        end
        # Update Loss Vectors
        push!(val_losses, epoch_val_loss/numobs(val_loader))
        # Log
        verbosity > 1 && @info(string("Loss after epoch ", epoch, 
            ":\n- Training: ", train_losses[end], "\n- Validation: ", val_losses[end]))
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