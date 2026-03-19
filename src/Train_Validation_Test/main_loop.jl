"""
`save_model(tstate, config)`

Persists the final training state to disk using JLD2.

# Arguments
- `tstate`: The training state returned by `epoch_loop`, containing model weights,
  optimizer state, and RNG state.
- `config`: Configuration dictionary. Recognised keys under `config["model"]`:
  - `"save_path"` : destination file path (default: `"model_checkpoint.jld2"`).
	The `.jld2` extension is appended automatically if absent.

# Returns
- The absolute path of the file that was written.

# Reload example
```julia
tstate = JLD2.load("model_checkpoint.jld2", "tstate")
```

# Notes
- JLD2 serialises the full `tstate` struct natively — model weights, optimizer
  state, and RNG state are all preserved without any custom serialisation logic.
- The parent directory is created automatically if it does not exist.
- A `"saved_at"` timestamp and the `config` snapshot are stored alongside
  `tstate` in the same file for reproducibility.
"""
function save_model(tstate, config)
	save_path = get(get(config, "model", Dict()), "save_path", "model_checkpoint.jld2")

	# Ensure the file has the correct extension
	if !endswith(save_path, ".jld2")
		save_path = save_path * ".jld2"
	end

	# Create parent directory if it does not exist
	parent_dir = dirname(save_path)
	if !isempty(parent_dir) && !isdir(parent_dir)
		mkpath(parent_dir)
		println("Created directory: $parent_dir")
	end

	println("Saving model state to $save_path")
	JLD2.jldsave(save_path;
		tstate   = tstate,
		config   = config,
		saved_at = string(Dates.now()),
	)

	abs_path = abspath(save_path)
	println("Model state saved successfully to $abs_path")
	return abs_path
end


"""
`main_loop(hdf5_path, config_path, rng_seed, loss_function_custom = nothing)`

The main driver function to orchestrate the training process for a segmentation model
using configurations and data from an HDF5 file.

# Arguments
- `hdf5_path`: Path to the HDF5 file containing the dataset.
- `config_path`: Path to the JSON configuration file that defines model and training
  parameters.
- `rng_seed`: Seed for the random number generator to ensure reproducibility.
- `loss_function_custom`: Optional custom loss function to override the one specified
  in the configuration file.

# Returns
- The trained model's state after completing the training process.

# Description
This function initialises the training environment, loads the data, and iteratively
trains a segmentation model using specified hyperparameters and optimisation strategies.
It supports cross-validation, early stopping, and can optionally utilise a custom loss
function if provided.

# Errors
- Raises errors related to file access, data integrity issues, or if essential
  configuration parameters are missing.
"""
function main_loop(hdf5_path, config_path, rng_seed, loss_function_custom = nothing)
    function main(h5, config_path, rng_seed)
        rng = Xoshiro(rng_seed)
        println("Loading configuration from $config_path")
        config = JSON.parsefile(config_path)
        println("Loading data from HDF5")

        indices_dict = proc_hdf5(h5, config, rng)

        image_data        = read(h5[indices_dict["train"][1] * "/images/data"])
        train_groups      = indices_dict["train"]
        validation_groups = indices_dict["validation"]
        unique_classes    = get_class_labels(indices_dict["train"], h5, config)
        num_classes = length(unique_classes) + 1 # TODO: work in progress - for validations
        model       = create_segmentation_model(num_classes, size(image_data, 4))
        optimizer   = get_optimiser(config["model"]["optimizer_name"])

        loss_function = if loss_function_custom !== nothing
            loss_function_custom
        else
            get_loss_function(config["model"]["loss_function_name"])
        end

        num_epochs_val = get(config["model"], "num_epochs", 10)  # default 10
        num_epochs     = isnothing(num_epochs_val) ? 10 : parse(Int, string(num_epochs_val))

        use_gpu = (config["augmentation"]["processing_unit"] == "GPU")
        tstate  = initialize_train_state(rng, model, optimizer; use_gpu = use_gpu)

        # TODO: add already tested apply.jl (requires the addition of probabilistic augmentation)
        final_tstate = if get(config["learning"], "n_cross_val", false)
            n_folds          = get(config["learning"], "n_folds", false)
            all_tstate       = []
            combined_indices = [indices_dict["train"]; indices_dict["validation"]]
            shuffled_indices = shuffle(rng, combined_indices)

            local last_tstate = tstate
            for fold in 1:n_folds
                println("Starting fold $fold/$n_folds")
                train_groups, validation_groups = k_fold_split(
                    shuffled_indices,
                    n_folds,
                    fold,
                    rng,
                )

                fold_tstate = initialize_train_state(rng, model, optimizer; use_gpu = use_gpu)
                fold_tstate = epoch_loop(
                    num_epochs,
                    train_groups,
                    validation_groups,
                    h5,
                    model,
                    fold_tstate,
                    config,
                    loss_function,
                    num_classes,
                )

                push!(all_tstate, fold_tstate)
                last_tstate = fold_tstate
            end
            last_tstate
        else
            epoch_loop(
                num_epochs,
                train_groups,
                validation_groups,
                h5,
                model,
                tstate,
                config,
                loss_function,
                num_classes,
            )
        end

        # TODO: add the already tested write function
        return final_tstate
    end

    h5open(hdf5_path, "r") do h5
        return main(h5, config_path, rng_seed)
    end
end


"""
`epoch_loop(num_epochs, group_paths_train, group_paths_val, h5, model, tstate, config,
			loss_function, num_classes)`

Executes the training and validation loop for a specified number of epochs.

# Arguments
- `num_epochs`: The total number of epochs to train the model.
- `group_paths_train`: Paths to training groups within the HDF5 dataset.
- `group_paths_val`: Paths to validation groups within the HDF5 dataset.
- `h5`: HDF5 file handle containing the dataset.
- `model`: The machine learning model to be trained.
- `tstate`: Training state containing optimiser and potentially other training-related
  parameters.
- `config`: Configuration dictionary specifying training options and parameters.
- `loss_function`: Loss function to be used for training.
- `num_classes`: Number of classes in the dataset including the background.

# Returns
- The updated training state after completing the training and validation cycles.

# Description
Conducts training by iterating over the specified number of epochs, applying the loss
function, and updating the model weights. Evaluates the model on the validation dataset
after every epoch, using the metric defined in the configuration. Supports early stopping
based on validation performance to prevent overfitting.
"""
function epoch_loop(num_epochs, group_paths_train, group_paths_val, h5, model, tstate,
	config, loss_function, num_classes)

    get(config["model"], "early_stopping", false) ?
        early_stopping_dict = Dict(
            "best_metric" => Inf,
            "patience_counter" => 0,
            "stop_training" => false,
        ) : nothing

    for epoch in 1:num_epochs
        # Training
        if get(config["model"], "early_stopping", false)
            println(
                "..................Starting epoch $epoch with early stopping ........................",
            )
            tstate, early_stopping_dict = train_epoch(
                group_paths_train,
                group_paths_val,
                h5,
                model,
                tstate,
                config,
                loss_function,
                num_classes,
                early_stopping_dict,
            )
            if early_stopping_dict["stop_training"]
                println("Stopping training early at epoch $epoch.")
                break
            end
        else
            println("..................Starting epoch $epoch ........................")
            tstate = train_epoch(
                group_paths_train,
                group_paths_val,
                h5,
                model,
                tstate,
                config,
                loss_function,
                num_classes,
            )
        end

        # Validation
        if !isempty(group_paths_val)
            val_metric = evaluate_validation(
                group_paths_val,
                h5,
                model,
                tstate,
                loss_function,
                config,
                num_classes,
            ) # TODO: work in progress
            println("Epoch $epoch, Validation Metric: $val_metric")
        end
    end
    return tstate
end
