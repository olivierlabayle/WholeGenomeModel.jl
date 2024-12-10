
mutable struct DLWholeGenomeRegression 
    model
    optimiser
    batch_size::Int
    max_epochs::Int
    rng
    device
end

function make_whole_genome_model(X, model_type=MLP; 
    optimiser=Adam(), 
    rng=Random.default_rng(), 
    device=cpu_device(), 
    max_epochs=100, 
    batch_size=8,
    model_kwargs...
    )
    input_size = size(X[[1]], 1)
    model = model_type(input_size; model_kwargs...)
    return DLWholeGenomeRegression(model, optimiser, batch_size, max_epochs, rng, device)
end

function fit(model::DLWholeGenomeRegression, verbosity, X, y)
    ps, st = Lux.setup(model.rng, model.model) |> model.device
    train_state = Lux.Training.TrainState(model.model, ps, st, model.optimiser)
    train_dataloader = DataLoader((X=X, y=y), batchsize=model.batch_size, rng=model.rng, shuffle=true)
    for epoch in 1:model.max_epochs
        epoch_loss = 0.
        for (x, y) in model.device(train_dataloader)
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

