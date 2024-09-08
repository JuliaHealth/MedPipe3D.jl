using Random
using MedPipe3D
using HDF5
using Lux
using JLD2,ImageMagick,TensorBoardLogger


using after_preprocess.get_batch
using after_preprocess.training
using after_preprocess.validation
using after_preprocess.testing
using after_preprocess.OptimiserSelector


"""
Performs the training and evaluation loop for a specified number of epochs.

# Arguments
- `num_epochs`: The number of epochs to train the model.
- `train_indices`: Indices for the training data.
- `val_indices`: Indices for the validation data.
- `model`: The model to be trained and evaluated.
- `tstate`: The current training state of the model.
- `config`: Configuration object containing training parameters.
- `loss_function`: The loss function to be used for training.
- `best_val_metric`: The best validation metric observed so far.
- `patience_counter`: Counter for early stopping patience.

# Returns
- The updated training state after the epoch loop.
"""
function epoch_loop(num_epochs, train_indices, val_indices, model, tstate, config, loss_function, best_val_metric,logger)

    patience_counter = 0
    for epoch in 1:num_epochs
        # Train the model
        tstate, train_metrics = train_epoch(train_indices, model, tstate, 1, config, loss_function,logger)
        
        # Evaluate on validation set
        val_metrics = evaluate_validation(val_indices, model, tstate, config,logger)

        current_val_metric = mean(val_metrics)
        if current_val_metric < best_val_metric
            best_val_metric = current_val_metric
            patience_counter = 0
        else                
            # Saving the model if the validation metric is currently the best one
            @save config.checkpoint_path tstate.parameters, tstate.states

            patience_counter += 1
        end

        # Early stopping check
        if config.early_stopping
            if patience_counter >= config.patience
                println("Early stopping at epoch $epoch")
                break
            end
        end
    end

    return tstate
end

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
    
    # Seeding
    rng = Xoshiro(0)
    # Initialize logger
    logger = TBLogger(config.logger_path, min_level=Logging.Info)

    #get optimiser
    opt = get_optimiser(config.optimizer, config_optimiser_args)
    num_epochs=config.num_epochs

    # Process HDF5 to get indices
    h5=HDF5.open(db_path)
    indices_dict = process_hdf5(h5, config, rng)    
    # Initialize training state
    tstate = initialize_train_state(rng, model, opt)
    #loading state from path if indicated so by config adapted from https://lux.csail.mit.edu/dev/tutorials/beginner/3_SimpleRNN#Saving-the-Model
    if(config.load_state)
        @load config.checkpoint_path tstate.parameters, tstate.states
    end

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
            
            epoch_loop(num_epochs, train_indices, val_indices, model, tstate, config, loss_function, best_val_metric,logger)

        end
    else
        # No folds, use the provided indices directly
        train_indices = indices_dict["0"]["training"]
        val_indices = indices_dict["0"]["validation"]
        
        epoch_loop(num_epochs, train_indices, val_indices, model, tstate, config, loss_function, best_val_metric,logger)
        
    end
    
    test_indices = indices_dict["test"]
    # Evaluate on test set
    test_metrics = evaluate_test_set(test_indices, model, tstate, test_time_augs, config, metadata_ref,logger)


end

