module training
using Lux
"""
Initializes the training state for the model.

# Arguments
- `rng`: Random number generator.
- `model`: The model to be trained.
- `opt`: The optimizer to be used for training.

# Returns
- The initialized training state, moved to the GPU.
"""
function initialize_train_state(rng, model, opt)
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    return cu(tstate)
end

"""
Trains the model for a specified number of epochs.

# Arguments
- `train_indices`: Indices of the training data.
- `model`: The model to be trained.
- `tstate`: The current training state of the model.
- `num_epochs`: The number of epochs to train the model.

# Returns
- The updated training state after training.
"""
function train_epoch(train_indices, model, tstate, num_epochs,config,loss_function)
        train_metric = []

        for train_index_sublist in train_indices
            data, label = fetch_and_preprocess_data(train_index_sublist)
            data = augment(data)
            if  (config.device=="CUDA")
                data = cu(data) # Move data to GPU
                label = cu(label) # Move data to GPU
            end
            _, loss, _, tstate = Lux.Experimental.single_train_step!(
                Lux.Experimental.ADTypes.AutoZygote(),
                loss_function,
                CuArray(data[train_index]), label,
                tstate)

            if config.log_train || config.early_stopping
                train_metrics = []
                for train_index_sublist in train_indices
                    data, label = fetch_and_preprocess_data(train_index_sublist)
                    data = augment(data)
                    if config.device == "CUDA"
                        data = cu(data) # Move data to GPU
                        label = cu(label) # Move data to GPU
                    end
                    y_pred, st = infer_model(tstate, model, data)
                    metric = evaluate_metric(y_pred, label, config)
                    push!(train_metrics, metric)
                end
                
                
                # Log the training metric if to_log is true

            end


        end

        if (length(train_metrics) == 0)
            train_metric = []
            else
            train_metric = mean(train_metrics)
            if config.to_log_train
                log_metric(logger, "train_metric", train_metric, epoch)
            end
        end    

    return tstate,train_metric
end

end


