"""
`initialize_train_state(rng, model, optimizer)`

Initializes the training state, which includes the random number generator, model, and optimizer.

# Arguments
- `rng`: The random number generator for reproducibility.
- `model`: The machine learning model to be trained.
- `optimizer`: The optimizer to be used in training.

# Returns
- The initialized training state encapsulating the RNG, model, and optimizer.
"""
function initialize_train_state(rng, model, optimizer)
    tstate = TrainState(rng, model, optimizer)
    return tstate
end


"""
`check_early_stopping(current_metric, best_metric, config, counter)`

Checks whether early stopping criteria are met based on improvement in the specified metric.

# Arguments
- `current_metric`: The current epoch's performance metric.
- `best_metric`: The best performance metric observed so far.
- `config`: Configuration dict containing early stopping parameters.
- `counter`: The current count of epochs without significant improvement.

# Returns
- Updated best metric, counter, and a boolean indicating whether to stop training.

# Description
Updates the best metric if current performance is better; increments the patience counter otherwise. Stops training if improvements are not observed for a configured number of epochs.
"""
function check_early_stopping(current_metric, best_metric, config, counter)
    early_stopping_min_delta = config["model"]["early_stopping_min_delta"]
    if isempty(current_metric)
        return best_metric, counter, false
    end
    improvement = best_metric - current_metric
    println("Improvement: $improvement")
    println(typeof(improvement))
    if improvement > early_stopping_min_delta
        best_metric = current_metric
        counter = 0
        println("Improvement found: New best metric is $best_metric")
    else
        counter += 1
        println("No improvement. Patience counter: $counter")
    end

    if counter >= config["model"]["patience"]
        return best_metric, counter, true
    end

    return best_metric, counter, false
end


"""
`train_epoch(train_group_paths, validation_group_paths, h5, model, tstate, config, loss_function, num_classes, early_stopping_dict = nothing)`

Conducts training and validation for one epoch, applying early stopping if configured.

# Arguments
- `train_group_paths`: Paths to training data groups.
- `validation_group_paths`: Paths to validation data groups.
- `h5`: HDF5 file handle for data access.
- `model`: The model to be trained.
- `tstate`: Current state of the training including model weights.
- `config`: Training and model configuration settings.
- `loss_function`: The loss function for evaluating training performance.
- `num_classes`: Number of classes in the dataset.
- `early_stopping_dict`: Dictionary containing early stopping information, if any.

# Returns
- Updated training state, optionally including updated early stopping information.

# Description
Performs training operations by processing batches of data, evaluating performance, and applying the optimizer. Optionally evaluates early stopping criteria based on validation loss or another configured metric.
"""
function train_epoch(tain_group_paths, validation_group_paths, h5, model, tstate, config, loss_function, num_classes, early_stopping_dict = nothing)
    train_metrics = []
    batch_size = config["data"]["batch_size"]
    num_batches = ceil(Int, length(tain_group_paths) / batch_size)
    early_stopping_metric = config["model"]["early_stopping_metric"]
    early_stopping_min_delta = config["model"]["early_stopping_min_delta"]
    metric = config["learning"]["metric"]
    
    if config["model"]["early_stopping"]
        best_metric = early_stopping_dict["best_metric"]
        patience_counter = early_stopping_dict["patience_counter"]
        stop_training = early_stopping_dict["stop_training"]
    end

    for batch_idx in 1:num_batches
        batch_paths = tain_group_paths[(batch_idx - 1) * batch_size + 1:min(batch_idx * batch_size, length(tain_group_paths))]
        data, labels, unique_classes = fetch_and_preprocess_data(batch_paths, h5, config)
        _, loss, _, tstate = Lux.Training.single_train_step!(Lux.Experimental.ADTypes.AutoZygote(), loss_function, (data, labels), tstate)
        y_pred, _ = infer_model(tstate, model, data)      
        push!(train_metrics, loss)
        println("Batch $batch_idx, Traing loss: $loss")

        if config["model"]["early_stopping"]
            # Calculate early stopping metric
            mean_val_metric, mean_val_loss  = evaluate_validation(validation_group_paths, h5, model, tstate, loss_function, config, num_classes)
            y_pred, _ = infer_model(tstate, model, data)
            #TODO: co jest sens tu logować
            #println("Batch $batch_idx, Training Loss: $loss, Metric: $metri")
            if early_stopping_metric == "val_loss"
                current_val_metric = mean_val_loss
            elseif early_stopping_metric == "metric"
                current_val_metric = mean_val_metric
            else
                error("Unsupported early stopping metric: $early_stopping_metric")
            end
            best_metric, patience_counter, stop_training = check_early_stopping(current_val_metric, best_metric, config, patience_counter)
            if stop_training
                println("Metric $early_stopping_metric last best value: $best_metric after $patience_counter batches with improvement less then $early_stopping_min_delta.")
                break
            end
        end
    end
    if config["model"]["early_stopping"]
        early_stopping_dict["best_metric"] = best_metric
        early_stopping_dict["patience_counter"] = patience_counter
        early_stopping_dict["stop_training"] = stop_training
        return tstate, early_stopping_dict
    else
        return tstate
    end
end