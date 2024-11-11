"""
`main_loop(hdf5_path, config_path, rng_seed, loss_function_custom = nothing)`

The main driver function to orchestrate the training process for a segmentation model using configurations and data from an HDF5 file.

# Arguments
- `hdf5_path`: Path to the HDF5 file containing the dataset.
- `config_path`: Path to the JSON configuration file that defines model and training parameters.
- `rng_seed`: Seed for the random number generator to ensure reproducibility.
- `loss_function_custom`: Optional custom loss function to override the one specified in the configuration file.

# Returns
- The trained model's state after completing the training process.

# Description
This function initializes the training environment, loads the data, and iteratively trains a segmentation model using specified hyperparameters and optimization strategies.
It supports cross-validation, early stopping, and can optionally utilize a custom loss function if provided.

# Errors
- Raises errors related to file access, data integrity issues, or if essential configuration parameters are missing.
"""
function main_loop(hdf5_path, config_path, rng_seed, loss_function_custom = nothing)
    function main(h5, config_path, rng_seed)
        rng = Xoshiro(rng_seed)
        println("Loading configuration from $config_path")
        config = JSON.parsefile(config_path)
        println("Loading data from HDF5")

        indices_dict = proc_hdf5(h5, config, rng)

        image_data = read(h5[indices_dict["train"][1] * "/images/data"])
        train_groups = indices_dict["train"]
        validation_groups = indices_dict["validation"]
        unique_classes = get_class_labels(indices_dict["train"], h5, config)
        num_classes = length(unique_classes) + 1 #TODO: work in progres - for validations
        model = create_segmentation_model(num_classes, size(image_data, 4))      
        optimizer = get_optimiser(config["model"]["optimizer_name"])
        if loss_function_custom !== nothing
            loss_function = loss_function_custom
        else
            loss_function = get_loss_function(config["model"]["loss_function_name"])
        end
        num_epochs = config["model"]["num_epochs"]

        tstate = initialize_train_state(rng, model, optimizer)
        #TODO: add already tested apply.jl (requires the addition of probabilistic augmentation)
        if config["learning"]["n_cross_val"]
            n_folds = config["learning"]["n_folds"]
            all_tstate = []
            combined_indices = [indices_dict["train"]; indices_dict["validation"]]
            shuffled_indices = shuffle(rng, combined_indices)
            for fold in 1:n_folds
                println("Starting fold $fold/$n_folds")
                train_groups, validation_groups = k_fold_split(shuffled_indices, n_folds, fold, rng)
                
                tstate = initialize_train_state(rng, model, optimizer)
                final_tstate = epoch_loop(num_epochs, train_groups, validation_groups, h5, model, tstate, config, loss_function, num_classes)
                
                push!(all_tstate, final_tstate)
            end
        else
            final_tstate = epoch_loop(num_epochs, train_groups, validation_groups, h5, model, tstate, config, loss_function, num_classes)
        end
    #TODO: add the already tested write function
        return final_tstate
    end
    h5open(hdf5_path, "r") do h5
        return main(h5, config_path, rng_seed)
    end
end


"""
`epoch_loop(num_epochs, group_paths_train, group_paths_val, h5, model, tstate, config, loss_function, num_classes)`

Executes the training and validation loop for a specified number of epochs.

# Arguments
- `num_epochs`: The total number of epochs to train the model.
- `group_paths_train`: Paths to training groups within the HDF5 dataset.
- `group_paths_val`: Paths to validation groups within the HDF5 dataset.
- `h5`: HDF5 file handle containing the dataset.
- `model`: The machine learning model to be trained.
- `tstate`: Training state containing optimizer and potentially other training-related parameters.
- `config`: Configuration dictionary specifying training options and parameters.
- `loss_function`: Loss function to be used for training.
- `num_classes`: Number of classes in the dataset including the background.

# Returns
- The updated training state after completing the training and validation cycles.

# Description
This function conducts training by iterating over the specified number of epochs, applying the loss function, and updating the model weights.
It evaluates the model on the validation dataset periodically, using metrics defined in the configuration. Supports early stopping based on validation performance to prevent overfitting.
"""
function epoch_loop(num_epochs, group_paths_train, group_paths_val, h5, model, tstate, config, loss_function, num_classes)
    
    config["model"]["early_stopping"] ? early_stopping_dict = Dict("best_metric" => Inf, "patience_counter" => 0, "stop_training" => false) : nothing
    
    for epoch in 1:num_epochs
    # Training
        if config["model"]["early_stopping"]
            println("..................Starting epoch $epoch with early stopping ........................")
            tstate, early_stopping_dict = train_epoch(group_paths_train, group_paths_val, h5, model, tstate, config, loss_function, num_classes, early_stopping_dict)
            if early_stopping_dict["stop_training"]
                println("Stopping training early at epoch $epoch.")
                break
            end
        else
            println("..................Starting epoch $epoch ........................")
            tstate = train_epoch(group_paths_train, group_paths_val, h5, model, tstate, config, loss_function, num_classes)
        end
        # Validation
        if !isempty(group_paths_val)
            val_metric = evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes) #TODO: work in progres
            println("Epoch $epoch, Validation Metric: $val_metric")
        end
    end
    return tstate
end

