##################### Testy configuration #####################
#region testy configuracji
include("./Configuration/configuration.jl")
create_config_extended("C:\\MedPipe\\MedPipe3D.jl\\src\\Tests_autput\\Testy_config")

# Creation of defoliant cofiguration
## ✅ when no file exists
## ✅ gdy istnieje plik zostaje nadpisany
## ✅ plik o nowej nazwie

#endregion




























# Uruchomienie funkcji

#region testy get_batch
function test_patch_extraction()
    hdf5_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\batched_data.h5"
    config_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\test.json"
    rng_seed = 1234  # Example seed for reproducibility

    h5open(hdf5_path, "r") do file
        # Seed the random number generator for reproducibility
        rng = Xoshiro(rng_seed)
        config = JSON.parsefile(config_path)
        indices_dict = proc_hdf5(file, config, rng)
        images, labels, class = get_batch_with_classes(indices_dict["train"], file, config)
        println("Fetching and preprocessing data...")
        println("Extracted patch images: ", size(images), ", labels: ", size(labels))
    end
end

#test_patch_extraction()


function test_early_stopping_integration()
    hdf5_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\batched_data.h5"
    config_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\test.json"
    rng_seed = 1234  # Set a seed for reproducibility

    h5open(hdf5_path, "r") do h5
        rng = Xoshiro(rng_seed)
        config = JSON.parsefile(config_path)

        indices_dict = proc_hdf5(h5, config, rng)
        image_data = read(h5[indices_dict["train"][1] * "/images/data"])
        unique_classes = get_class_labels(indices_dict["train"], h5, config)
        num_classes = length(unique_classes) +1 # Add one for background class
        model = create_segmentation_model(num_classes, size(image_data, 4))

        optimizer = get_optimiser(config["model"]["optimizer_name"])
        loss_function = get_loss_function(config["model"]["loss_function_name"])
        num_epochs = config["model"]["num_epochs"]

        tstate = initialize_train_state(rng, model, optimizer)

        group_paths_train = indices_dict["train"]
        group_paths_val = indices_dict["validation"]

        final_tstate = epoch_loop(num_epochs, group_paths_train, group_paths_val, h5, model, tstate, config, loss_function, num_classes)
        println("Early stopping test completed.")
    end
end

function test_evaluate_validation()
    # Define paths
    hdf5_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\batched_data.h5"
    config_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\test.json"
    rng_seed = 1234  # Example seed for reproducibility

    # Open the HDF5 file and run the validation test
    h5open(hdf5_path, "r") do h5
        rng = Xoshiro(rng_seed)
        config = JSON.parsefile(config_path)
        indices_dict = proc_hdf5(h5, config, rng)
        num_classes = length(indices_dict["train"]) +1 # Add one for background class
        model = create_segmentation_model(num_classes, size(read(h5[indices_dict["train"][1] * "/images/data"]), 4))
        optimizer = get_optimiser(config["model"]["optimizer_name"])
        loss_function = get_loss_function(config["model"]["loss_function_name"])
        tstate = initialize_train_state(rng, model, optimizer)

        println("Running validation test...")
        evaluate_validation(indices_dict["validation"], h5, model, tstate, config, loss_function, num_classes)
    end
end

function save_results_test(y_pred, attributes, config)
    println("Saving results...")
    output_folder = get(config, "output_folder", "output")
    isdir(output_folder) || mkpath(output_folder)
    process_and_save_medimage_test(attributes, y_pred, output_folder, "_prediction")
end

function process_and_save_medimage_test(meta, data, output_folder, suffix)
    println("Processing and saving MedImage...")
    original_file_path = meta["file_path"]
    original_image = load_images(original_file_path)[1]  # Load the original MedImage

    # Ensure data dimensions match the original image's voxel data dimensions
    if size(data) != size(original_image.voxel_data)
        data = resize(data, size(original_image.voxel_data))
    end

    updated_image = update_voxel_and_spatial_data(
        original_image,
        data,
        original_image.spacing,
        original_image.origin,
        original_image.direction
    )

    filename_without_ext, ext = splitext(basename(original_file_path))
    new_filename = filename_without_ext * suffix * ext
    output_file_path = joinpath(output_folder, new_filename)

    create_nii_from_medimage(updated_image, output_file_path)
    println("Saved updated data to: $output_file_path")
end



function main_loop_test_1(hdf5_path, config_path, rng_seed)
    function main(h5, config_path, rng_seed)
        rng = Xoshiro(rng_seed)
        println("Loading configuration from $config_path")
        config = JSON.parsefile(config_path)
        println("Loading data from HDF5")

        indices_dict = proc_hdf5(h5, config, rng)

        image_data = read(h5[indices_dict["train"][1] * "/images/data"])
        train_groups = indices_dict["train"]
        validation_groups = indices_dict["validation"]
        test_groups = indices_dict["test"]
        
        unique_classes = get_class_labels(indices_dict["train"], h5, config)
        println("unique_classes: ", unique_classes)
        num_classes = length(unique_classes) +1
        model = create_segmentation_model(length(unique_classes), size(image_data, 4))      

        optimizer = get_optimiser(config["model"]["optimizer_name"])
        loss_function = get_loss_function(config["model"]["loss_function_name"])
        num_epochs = config["model"]["num_epochs"]

        tstate = initialize_train_state(rng, model, optimizer)

        final_tstate = epoch_loop(num_epochs, train_groups, validation_groups, h5, model, tstate, config, loss_function, num_classes)
        close(h5)
        return test_groups, model, final_tstate, config
    end
    h5open(hdf5_path, "r") do h5
        return main(h5, config_path, rng_seed)
    end
end

function main_loop_test_2(test_groups, hdf5_path, model, final_tstate, config)
    # After training, evaluate on test set
    function main(h5, config, test_groups, model, final_tstate)
        test_metrics = evaluate_test_set_test(test_groups, h5, model, final_tstate, config)
        println("Test metrics: ", test_metrics)
    end
    h5open(hdf5_path, "r") do h5
        return main(h5, config, test_groups, model, final_tstate)
    end
    return final_tstate, test_metrics    
end

function test_main_loop_test(test_groups, model, final_tstate, config, hdf5_path)
    final_tstate, test_metrics = main_loop_test_2(test_groups, hdf5_path, model, final_tstate, config)
    println("Testing completed successfully.")
    println("Test metrics: ", test_metrics)
end

#create_config_extended("test_data/zubik/saving_folder", "test.json")
#batch_main("C:\\MedImage\\MedImages.jl\\test_data\\zubik\\Test_data_set3",
#"C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder",
#"C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\test.json")
#print_hdf5_contents("C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\batched_data.h5")
#test_main_loop()
#hdf5_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\batched_data.h5"
#config_path = "C:\\MedImage\\MedImages.jl\\test_data\\zubik\\saving_folder\\test.json"
#rng_seed = 1234
#test_groups, model, final_tstate, config =  main_loop_test_1(hdf5_path, config_path, rng_seed)
#test_main_loop_test(test_groups, model, final_tstate, config, hdf5_path)