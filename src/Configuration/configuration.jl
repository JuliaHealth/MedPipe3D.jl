function create_config_extended(save_path::String, config_name="config.jl")
    config = Dict()
    # I'm sorry for shity design patterns, I know its terribly stiff and unreadable
# Data Parameters
    println("Enter batch data parameters:")
    println("Enter the batch size (batch_size) [default: 4]:")
    batch_size_input = readline()
    batch_size = isempty(batch_size_input) ? 4 : parse(Int, batch_size_input)
    println("Should the batch be complete if not fully filled? (true/false): [default: false]")
    batch_complete_input = readline()
    batch_complete = isempty(batch_complete_input) ? false : parse(Bool, batch_complete_input)

    println("Enter the batch size (channel_size) [default: 4]:")
    channel_size_input = readline()
    channel_size = isempty(channel_size_input) ? 4 : parse(Int, channel_size_input)

    println("Resample to first image? (true/false): [default: false]")
    resample_to_target_input = readline()
    resample_to_target = isempty(resample_to_target_input) ? false : parse(Bool, resample_to_target_input)
    
    println("Resample spacing? (set/avg/median): [default: avg]")
    resample_to_spacing_input = readline()
    resample_to_spacing = isempty(resample_to_spacing_input) ? "avg" : resample_to_spacing_input
    if resample_to_spacing == "set"
        println("Enter target spacing (e.g., (1.0, 1.0, 1.0)): [default: (1.0, 1.0, 1.0)]")
        target_spacing_input = readline()
        target_spacing = isempty(target_spacing_input) ? (1.0, 1.0, 1.0) : string_to_tuple(target_spacing_input)
    else
        target_spacing = nothing
    end

    println("Should the images be resize using average per channel or do you want to specify the size for all images? (avg/(x,y,z)): [default: avr]")
    resample_size_input = readline()
    resample_size = isempty(resample_size_input) ? "avg" : string_to_tuple(resample_size_input)
    

    println("Standardization? (true/false): [default: false]")
    standardization_input = readline()
    standardization = isempty(standardization_input) ? false : parse(Bool, standardization_input)

    println("Normalization? (true/false): [default: false]")
    normalization_input = readline()
    normalization = isempty(normalization_input) ? false : parse(Bool, normalization_input)

    data_params = Dict(
        "batch_size" => batch_size,
        "batch_complete" => batch_complete,
        "channel_size" => channel_size,
        "resample_to_target" => resample_to_target,
        "resample_to_spacing" => resample_to_spacing,
        "target_spacing" => target_spacing,
        "resample_size" => resample_size,
        "standardization" => standardization,
        "normalization" => normalization,
        "has_mask" => true
    )

    # Augmentation Parameters
    augmentations = [
        "Brightness transform",
        "Contrast augmentation transform",
        "Gamma transform",
        "Gaussian noise transform",
        "Rician noise transform",
        "Mirror transform",
        "Scale transform",
        "Gaussian blur transform",
        "Simulate low-resolution transform",
        "Elastic deformation transform"
    ]

    println("Select the augmentations you want to apply by entering their numbers separated by commas. The order will affect the processing sequence.")
    for (index, aug) in enumerate(augmentations)
        println("$(index). $aug")
    end

    selected_indices_input = readline()
    selected_indices = parse.(Int, split(selected_indices_input, ","))
    selected_order = [augmentations[i] for i in selected_indices]

# Collect Augmentation-Specific Parameters
    aug_params = Dict()
    for idx in selected_indices
        aug_name = augmentations[idx]
        println("Configuring $aug_name:")
        aug_config = Dict()

        # Based on the augmentation, collect specific parameters
        if aug_name == "Brightness transform"
            println("Enter value for 'value' [default: 0.2]:")
            value_input = readline()
            value = isempty(value_input) ? 0.2 : parse(Float32, value_input)

            println("Enter mode ('additive'/'multiplicative') [default: 'additive']:")
            mode_input = readline()
            mode = isempty(mode_input) ? "additive" : mode_input

            aug_config["value"] = value
            aug_config["mode"] = mode

        elseif aug_name == "Contrast augmentation transform"
            println("Enter factor for contrast [default: 1.5]:")
            factor_input = readline()
            factor = isempty(factor_input) ? 1.5 : parse(Float32, factor_input)

            aug_config["factor"] = factor

        elseif aug_name == "Gamma transform"
            println("Enter gamma value [default: 2.0]:")
            gamma_input = readline()
            gamma = isempty(gamma_input) ? 2.0 : parse(Float32, gamma_input)

            aug_config["gamma"] = gamma

        elseif aug_name == "Gaussian noise transform"
            println("Enter variance for Gaussian noise [default: 0.01]:")
            variance_input = readline()
            variance = isempty(variance_input) ? 0.01 : parse(Float32, variance_input)

            aug_config["variance"] = variance

        elseif aug_name == "Rician noise transform"
            println("Enter variance for Rician noise [default: 0.01]:")
            variance_input = readline()
            variance = isempty(variance_input) ? 0.01 : parse(Float32, variance_input)

            aug_config["variance"] = variance

        elseif aug_name == "Mirror transform"
            println("Enter axes to mirror (e.g., (1,2,3)) [default: (1,2,3)]:")
            axes_input = readline()
            axes = isempty(axes_input) ? (1,2,3) : eval(Meta.parse(axes_input))

            aug_config["axes"] = axes

        elseif aug_name == "Scale transform"
            println("Enter scale factor [default: 1.0]:")
            scale_factor_input = readline()
            scale_factor = isempty(scale_factor_input) ? 1.0 : parse(Float32, scale_factor_input)

            println("Enter interpolator enum (e.g., 'Linear_en') [default: 'Linear_en']:")
            interpolator_enum_input = readline()
            interpolator_enum = isempty(interpolator_enum_input) ? "Linear_en" : interpolator_enum_input

            aug_config["scale_factor"] = scale_factor
            aug_config["interpolator_enum"] = interpolator_enum

        elseif aug_name == "Gaussian blur transform"
            println("Enter sigma [default: 1.0]:")
            sigma_input = readline()
            sigma = isempty(sigma_input) ? 1.0 : parse(Float32, sigma_input)

            println("Enter kernel size [default: 5]:")
            kernel_size_input = readline()
            kernel_size = isempty(kernel_size_input) ? 5 : parse(Int, kernel_size_input)

            println("Enter shape ('2D'/'3D') [default: '3D']:")
            shape_input = readline()
            shape = isempty(shape_input) ? "3D" : shape_input

            println("Enter processing unit ('GPU'/'CPU') [default: 'GPU']:")
            processing_unit_input = readline()
            processing_unit_aug = isempty(processing_unit_input) ? "GPU" : processing_unit_input

            aug_config["sigma"] = sigma
            aug_config["kernel_size"] = kernel_size
            aug_config["shape"] = shape
            aug_config["processing_unit"] = processing_unit_aug

        elseif aug_name == "Simulate low-resolution transform"
            println("Enter blur sigma [default: 1.0]:")
            blur_sigma_input = readline()
            blur_sigma = isempty(blur_sigma_input) ? 1.0 : parse(Float32, blur_sigma_input)

            println("Enter kernel size [default: 5]:")
            kernel_size_input = readline()
            kernel_size = isempty(kernel_size_input) ? 5 : parse(Int, kernel_size_input)

            println("Enter downsample scale [default: 2.0]:")
            downsample_scale_input = readline()
            downsample_scale = isempty(downsample_scale_input) ? 2.0 : parse(Float32, downsample_scale_input)

            aug_config["blur_sigma"] = blur_sigma
            aug_config["kernel_size"] = kernel_size
            aug_config["downsample_scale"] = downsample_scale

        elseif aug_name == "Elastic deformation transform"
            println("Enter strength [default: 1.0]:")
            strength_input = readline()
            strength = isempty(strength_input) ? 1.0 : parse(Float32, strength_input)

            println("Enter interpolator enum (e.g., 'Linear_en') [default: 'Linear_en']:")
            interpolator_enum_input = readline()
            interpolator_enum = isempty(interpolator_enum_input) ? "Linear_en" : interpolator_enum_input

            aug_config["strength"] = strength
            aug_config["interpolator_enum"] = interpolator_enum

        end

        aug_params[aug_name] = aug_config
    end
    
# Collect p_rand and processing_unit
    println("Enter the probability of applying each augmentation (p_rand) [default: 0.5]:")
    p_rand_input = readline()
    p_rand = isempty(p_rand_input) ? 0.5 : parse(Float32, p_rand_input)

    println("Enter the processing unit (GPU/CPU) [default: GPU]:")
    processing_unit_input = readline()
    processing_unit = isempty(processing_unit_input) ? "GPU" : processing_unit_input

    augmentation_params = Dict(
        "order" => selected_order,
        "p_rand" => p_rand,
        "augmentations" => aug_params,
        "processing_unit" => processing_unit
    )

# Additional Pipeline Characteristics

    # Initialize variables to avoid UndefVarError
    Train_Val_Test_JSON = false
    class_JSON_path = false
    additional_JSON_path = false
    n_folds = 1
    n_lcc_input = nothing
    println("What metric for evaluation? (true/false): [default: dice]")
    metric_input = readline()
    metric = isempty(metric_input) ? "dice" : metric_input

    println("Do you whant to use largest connected component for validation and testing? (true/false): [default: false]")
    largest_connected_component_input = readline()
    largest_connected_component = isempty(largest_connected_component_input) ? false : parse(Bool, largest_connected_component_input)
    if largest_connected_component
        println("Enter the number of components: [default: 1]")
        n_lcc_input = readline()
        n_lcc = isempty(n_lcc_input) ? 1 : parse(Int, n_lcc_input)
    end

    println("Use n-fold cross-validation? (true/false): [default: false]")
    n_cross_val_input = readline()
    n_cross_val = isempty(n_cross_val_input) ? false : parse(Bool, n_cross_val_input)
    if n_cross_val
        println("Enter the number of folds (n): [default: 5]")
        n_folds_input = readline()
        n_folds = isempty(n_folds_input) ? 5 : parse(Int, n_folds_input)
    end

    println("Use probabilistic oversampling with patch data loading? (true/false): [default: false]")
    patch_probabilistic_oversampling_input = readline()
    patch_probabilistic_oversampling = isempty(patch_probabilistic_oversampling_input) ? false : patch_probabilistic_oversampling_input
    
    patch_size = nothing
    oversampling_probability = nothing
    if patch_probabilistic_oversampling == true
        println("Enter patch size (e.g., (64, 64, 64)): [default: (64, 64, 64)]")
        patch_size_input = readline()
        patch_size = isempty(patch_size_input) ? (64, 64, 64) : string_to_tuple(patch_size_input)
        
        println("Set the oversampling probability from 0, indicating completely random patches, to 1, where each patch will contain relevant information. : [default: 0.5]")
        oversampling_probability_input = readline()
        oversampling_probability = isempty(oversampling_probability_input) ? 0.5 : string_to_tuple(oversampling_probability_input)
    end


    println("Do you have specific collections in JSON: train, validation, test (false/path): [default: false]")
    is_test_train_validation_collections_input = readline()
    is_test_train_validation_collections = isempty(is_test_train_validation_collections_input) ? false : parse(Bool, is_test_train_validation_collections_input)
    if is_test_train_validation_collections
        println("Enter path to JSON for Traing, Validation and Test data")
        Train_Val_Test_JSON_input = readline()
        Train_Val_Test_JSON = isempty(Train_Val_Test_JSON_input) ? false : Train_Val_Test_JSON_input
    else
        println("Set test% train% validation% (e.g., 0.6, 0.2, 0.2): [default: 0.6, 0.2, 0.2]")
        test_train_validation_input = readline()
        test_train_validation = isempty(test_train_validation_input) ? (0.6, 0.2, 0.2) : test_train_validation_input
    end

    println("Do you have specific class collections (false/path): [default: false]")
    is_class_collections_input = readline()
    is_class_collections = isempty(is_class_collections_input) ? false : parse(Bool, is_class_collections_input)
    if is_class_collections
        println("Enter path to JSON for class data")
        class_JSON_path_input = readline()
        class_JSON_path = isempty(class_JSON_path_input) ? false : class_JSON_path_input
    end

    println("Invertible augmentations? (true/false): [default: false]")
    invertible_augmentations_input = readline()
    invertible_augmentations = isempty(invertible_augmentations_input) ? false : parse(Bool, invertible_augmentations_input)

    println("Should the channel be shuffle? (true/false): [default: false]")
    shuffle_input = readline()
    shuffle = isempty(shuffle_input) ? false : parse(Bool, shuffle_input)

    println("Do you have additional JSONs files (false/path): [default: false]")
    additional_JSON_input = readline()
    additional_JSON = isempty(additional_JSON_input) ? false : parse(Bool, additional_JSON_input)
    if additional_JSON
        println("Enter path to JSON for additional data")
        additional_JSON_path_input = readline()
        additional_JSON_path = isempty(additional_JSON_path_input) ? false : additional_JSON_path_input
    end
    
    learning_params = Dict(
        "invertible_augmentations" => invertible_augmentations,
        "Train_Val_Test_JSON" => Train_Val_Test_JSON,
        "n_cross_val" => n_cross_val,
        "n_folds" => n_folds,
        "test_train_validation" => test_train_validation,
        "class_JSON_path" => class_JSON_path,
        "additional_JSON_path" => additional_JSON_path,
        "shuffle" => shuffle,
        "patch_probabilistic_oversampling" => patch_probabilistic_oversampling,
        "patch_size" => patch_size,
        "metric" => metric,
        "oversampling_probability" => oversampling_probability,
        "largest_connected_component" => largest_connected_component,
        "n_lcc" => n_lcc
    )

# Model Parameters
    println("Enter model parameters:")
    println("Optimizer name (e.g., Adam): [default: Adam]")
    optimizer_name_input = readline()
    optimizer_name = isempty(optimizer_name_input) ? "Adam" : optimizer_name_input

    println("Loss function name (e.g., l1, Custom): [default: l1]")
    loss_function_name_input = readline()
    loss_function_name = isempty(loss_function_name_input) ? "l1" : loss_function_name_input

    if loss_function_name == "Custom"
        println("You need to use your loss function as argument in Main_loop.")
    end

    println("Optimizer arguments (e.g., lr=0.001): [default: lr=0.001]")
    optimizer_args_input = readline()
    optimizer_args = isempty(optimizer_args_input) ? "lr=0.001" : optimizer_args_input

    println("Number of epochs (num_epochs): [default: 50]")
    num_epochs_input = readline()
    num_epochs = isempty(num_epochs_input) ? 50 : parse(Int, num_epochs_input)

    println("Use early stopping? (true/false): [default: false]")
    early_stopping_input = readline()
    early_stopping = isempty(early_stopping_input) ? false : parse(Bool, early_stopping_input)

    if early_stopping
        println("Enter patience for early stopping: [default: 5]")
        patience_input = readline()
        patience = isempty(patience_input) ? 5 : parse(Int, patience_input)
        println("Enter min. delta for early stopping: [default: 0.001]")
        early_stopping_min_delta_input = readline()
        early_stopping_min_delta = isempty(early_stopping_min_delta_input) ? 0.001 : parse(Int, early_stopping_min_delta_input)
        println("Enter metric for early stopping: [default: val_loss]")
        early_stopping_metric_input = readline()
        early_stopping_metric = isempty(early_stopping_metric_input) ? val_loss : parse(Int, early_stopping_metric_input)
    else
        patience = nothing
        early_stopping_min_delta = nothing
        early_stopping_metric = nothing
    end

    model_params = Dict(
        "optimizer_name" => optimizer_name,
        "optimizer_args" => optimizer_args,
        "num_epochs" => num_epochs,
        "early_stopping" => early_stopping,
        "patience" => patience,
        "early_stopping_min_delta" => early_stopping_min_delta,
        "early_stopping_metric" => early_stopping_metric,
        "loss_function_name" => loss_function_name
    )

    # Combine all parameters into config
    config["data"] = data_params
    config["augmentation"] = augmentation_params
    config["learning"] = learning_params
    config["model"] = model_params

    # Save to JSON
    json_string = JSON.json(config, 4)

    # Save to file
    json_path = joinpath(save_path, config_name)
    open(json_path, "w") do file
        print(file, json_string)
    end
    println("Configuration saved to $json_path")
    return json_path
end


function modify_config(config::Dict{String, Any}, action::Symbol, path::Vector{String}, value=nothing)
    current = config
    for i in 1:length(path)-1
        key = path[i]
        if !haskey(current, key)
            if action == :add
                current[key] = Dict{String, Any}()
            else
                println("Key $(join(path[1:i], ".")) does not exist. Cannot proceed with action: $action.")
                return config
            end
        end
        current = current[key]
        if !(current isa Dict{String, Any})
            println("Cannot navigate through non-dictionary at path $(join(path[1:i], "."))")
            return config
        end
    end
    key = path[end]
    if action == :add
        if haskey(current, key)
            println("Warning: Key $(join(path, ".")) already exists. Use :modify action to change the value.")
        else
            current[key] = value
            println("Added key $(join(path, ".")) with value: $value")
        end
    elseif action == :modify
        if haskey(current, key)
            current[key] = value
            println("Modified key $(join(path, ".")) to value: $value")
        else
            println("Key $(join(path, ".")) does not exist. Cannot modify a non-existent key.")
        end
    elseif action == :remove
        if haskey(current, key)
            delete!(current, key)
            println("Removed key $(join(path, "."))")
        else
            println("Key $(join(path, ".")) not found. Cannot remove a non-existent key.")
        end
    else
        error("Unknown action: $action. Valid actions are :add, :remove, or :modify.")
    end
    return config
end