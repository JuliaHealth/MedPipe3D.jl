function initialize_train_state(rng, model, optimizer)
    tstate = TrainState(rng, model, optimizer)
    return tstate
end

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