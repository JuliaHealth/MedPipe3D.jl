module main_loop
using Random
using MedPipe3D
using HDF5Utils
using after_preprocess.get_batch
using after_preprocess.training
using after_preprocess.validation
using after_preprocess.testing
using after_preprocess.OptimiserSelector

"""
Main loop function to process HDF5, initialize training state, and perform training and evaluation.
# Arguments
- `db_path::String`: Path to the HDF5 database.
- `config::Configuration`: Configuration object.
- `rng::AbstractRNG`: Random number generator.
- `model`: The model to be trained and evaluated.
- `opt`: The optimizer to be used for training.
- `num_epochs::Int`: Number of epochs for training.
- `loss_function`: Loss function to be used during training.
- `test_time_augs`: Test time augmentations.
- `metadata_ref`: Metadata reference for evaluation.
"""
function main_loop(db_path::String, config::Configuration, rng::AbstractRNG, model, loss_function,  metadata_ref)
    

    opt = get_optimiser(config.optimizer)
    num_epochs=config.num_epochs
    test_time_augs=config.test_time_augs
    # Process HDF5 to get indices
    h5=HDF5.open(db_path)
    indices_dict = process_hdf5(h5, config, rng)
    
    # Initialize training state
    tstate = initialize_train_state(rng, model, opt)
    
    best_val_metric = Inf
    #for early stopping
    patience_counter = 0
    
    # Check for folds in the indices dictionary
    if haskey(indices_dict, "1")
        for i in 1:config.n_fold_cross_val
            fold_key = "$i"
            fold_groups = indices_dict[fold_key]
            train_indices = fold_groups["training"]
            val_indices = fold_groups["validation"]
            test_indices = fold_groups["test"]
            
            for epoch in 1:num_epochs
                # Train the model
                tstate, train_metrics = train_epoch(train_indices, model, tstate, 1, config, loss_function)
                
                # Evaluate on validation set
                val_metrics = evaluate_validation(val_indices, model, tstate, config)
                
                # Early stopping check
                if config.early_stopping
                    current_val_metric = mean(val_metrics)
                    if current_val_metric < best_val_metric
                        best_val_metric = current_val_metric
                        patience_counter = 0
                    else
                        patience_counter += 1
                    end
                    if patience_counter >= config.patience
                        println("Early stopping at epoch $epoch")
                        break
                    end
                end
            end
        end
    else
        # No folds, use the provided indices directly
        train_indices = indices_dict["0"]["training"]
        val_indices = indices_dict["0"]["validation"]
        
        for epoch in 1:num_epochs
            # Train the model
            tstate, train_metrics = train_epoch(train_indices, model, tstate, 1, config, loss_function)
            
            # Evaluate on validation set
            val_metrics = evaluate_validation(val_indices, model, tstate, config)
            
            # Early stopping check
            if config.early_stopping
                current_val_metric = mean(val_metrics)
                if current_val_metric < best_val_metric
                    best_val_metric = current_val_metric
                    patience_counter = 0
                else
                    patience_counter += 1
                end
                if patience_counter >= config.patience
                    println("Early stopping at epoch $epoch")
                    break
                end
            end
        end
    end
    
    test_indices = indices_dict["test"]
    # Evaluate on test set
    test_metrics = evaluate_test_set(test_indices, model, tstate, test_time_augs, config, metadata_ref)
end

end # module