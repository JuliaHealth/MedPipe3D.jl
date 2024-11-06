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
        num_classes = length(unique_classes) + 1 # Add 1 for background class
        model = create_segmentation_model(num_classes, size(image_data, 4))      
        optimizer = get_optimiser(config["model"]["optimizer_name"])
        if loss_function_custom !== nothing
            loss_function = loss_function_custom
        else
            loss_function = get_loss_function(config["model"]["loss_function_name"])
        end
        num_epochs = config["model"]["num_epochs"]

        tstate = initialize_train_state(rng, model, optimizer)
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
        return final_tstate
    end
    h5open(hdf5_path, "r") do h5
        return main(h5, config_path, rng_seed)
    end
end


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
#TODO: Validacja powina byćco epoch a czy co step/batch?
        # Validation
        if !isempty(group_paths_val)
            val_metric = evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes)
            println("Epoch $epoch, Validation Metric: $val_metric")
        end
    end
    return tstate
end

