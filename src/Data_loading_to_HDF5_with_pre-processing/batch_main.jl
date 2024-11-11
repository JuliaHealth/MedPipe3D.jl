"""
`batch_main(main_folder::String, save_path::String, config_path::String=nothing, config_name::String="config.json")`

Executes the main processing routine for batching and saving processed medical image and mask data.

# Arguments
- `main_folder`: The root directory containing patient data folders.
- `save_path`: The path where processed data will be saved.
- `config_path`: Optional path to an existing configuration JSON file. If not provided, a new one will be created.
- `config_name`: Name for a newly created configuration file if `config_path` is not provided.

# Returns
- Function does not explicitly return unless it successfully completes processing and saving data,
in which case it might return a confirmation or success message, depending on the implementation of `save_to_hdf5`.

# Description
The function first checks whether a configuration path is provided; if not, it creates a new configuration.
It then identifies and segregates image and mask folders within the patient directories, processes them, and finally, calls another function to save the processed data.
It ensures that both image and mask data are available for each patient, handling different channels and applying the configurations set forth in the JSON configuration file.

# Errors
Throws an error if the necessary image or mask directories are not found within the patient folders, or if any other critical step fails during processing.
"""

function batch_main(main_folder::String, save_path, config_path=nothing, config_name = "config.json")
    if config_path === nothing                                                  # Check if a specific configuration file path is provided, otherwise create a new configuration
        create_config_extended("test_data/zubik/saving_folder", config_name)
        config_path = "test_data/zubik/saving_folder/config.json"
    end
    image_channel_folders = String[]                                            # Check if a specific configuration file path is provided, otherwise create a new configuration
    mask_channel_folders = String[]
    patient_folders = readdir(main_folder)
    for patient_folder in patient_folders                                       # Search for subfolders named 'Images' and 'Masks' within each patient folder
        patient_path = joinpath(main_folder, patient_folder)
        if isdir(patient_path)
            subfolders = readdir(patient_path)
            for subfolder in subfolders
                subfolder_path = joinpath(patient_path, subfolder)
                if isdir(subfolder_path)
                    if contains(lowercase(subfolder), "image")
                        push!(image_channel_folders, subfolder_path)
                    elseif contains(lowercase(subfolder), "mask")
                        push!(mask_channel_folders, subfolder_path)
                    end
                end
            end
        end
    end
    #TODO: Change so that it is not 2 separate calls but 1 loop depending on the channel name
    if !isempty(image_channel_folders) && !isempty(mask_channel_folders)       # Search for subfolders named 'Images' and 'Masks' within each patient folder
        # Process images
        println("Processing image and mask folders in: $main_folder")
        println("\nProcessing images.")
        # Call functions to process images and save them
        images_batches, images_metadata, channel_names  = load_create_dataset_and_metadata(image_channel_folders, config_path, Linear_en, "image")

        # Process masks
        println("\nProcessing masks.")
        # Call functions to process masks and  save them
        masks_batches, masks_metadata  = load_create_dataset_and_metadata(mask_channel_folders, config_path, Nearest_neighbour_en,"mask")
        
        return save_to_hdf5(images_batches, images_metadata, masks_batches, masks_metadata, save_path, channel_names)
    else
        error("Could not find both image or mask folders per patient folder in the specified main folder.")
    end
end


"""
`load_create_dataset_and_metadata(channel_paths::Vector{String}, config_path::String, interpolator::Interpolator_enum, channel_type::String)`

Loads and processes medical image datasets from specified paths, applying configurations from a JSON file.

# Arguments
- `channel_paths`: A vector of directory paths containing the image or mask files.
- `config_path`: Path to the JSON configuration file which includes processing parameters.
- `interpolator`: An enum defining the type of interpolation to use during image resampling.
- `channel_type`: Specifies the type of data being processed, e.g., 'image' or 'masks'.

# Returns
- `Tuple`: A tuple containing the final tensor of processed images, metadata for each image, and the names of the channels.

# Description
This function processes medical images or masks by loading settings from a configuration file.
It handles different image channels, applies specified image processing steps like resampling, normalization, and standardization, and organizes images into a structured tensor format.
Metadata is also gathered and structured for each image, facilitating further data management or analysis.

# Errors
Throws an error if there are any inconsistencies with the expected data inputs or processing parameters. Errors are also raised for file mismatches, configuration issues, or if the processing steps fail due to invalid data properties.
"""

#TODO: This function should be separated into Loading and pre-processing(resample_to_image, resample_to_spacing, crop_or_pad)
function load_create_dataset_and_metadata(channel_paths::Vector{String}, config_path::String, interpolator::Interpolator_enum, channel_type::String)
    println("Loading configuration from: $config_path")
    config = JSON.parsefile(config_path)                                          # Load configuration settings from the specified JSON file
    if channel_type == "image"                                                    # Determine channel size based on whether images or masks are being processed
        channel_size = config["data"]["channel_size_imgs"]
    elseif channel_type == "masks"
        channel_size = config["data"]["channel_size_masks"]
    else 
        println("Images and masks are required.")
    end
#TODO: probably wrong or unnecessary
    batch_complete = config["data"]["batch_complete"]
    # Extract other important parameters from configuration
    resample_images_to_target = config["data"]["resample_to_target"]
    resample_images_spacing = config["data"]["resample_to_spacing"]
    target_spacing = config["data"]["target_spacing"]
    resample_size = config["data"]["resample_size"]
    standardization = config["data"]["standardization"]
    normalization = config["data"]["normalization"]

    # Optional JSON paths for splitting data and class mapping
    config["learning"]["Train_Val_Test_JSON"] != false ? dataset_splits = config["learning"]["Train_Val_Test_JSON"] : dataset_splits = false
    config["learning"]["class_JSON_path"] != false ? class_mapping = config["learning"]["class_JSON_path"] : class_mapping = false
    
    # Prepare to collect processed data and metadata
    channels_data = []
    all_metadata = []
    error_messages = []
    println("Processing $class_names.")
    channel_names = [basename(dirname(path)) for path in channel_paths]
    
    # Process each channel folder found in the given paths
    for channel_path in channel_paths
        channel_folder = basename(dirname(channel_path))
        println("Processing channel folder: $channel_folder")
        image_files = [joinpath(channel_path, file) for file in readdir(channel_path) if isfile(joinpath(channel_path, file))]
        needed_image_files = sort(image_files)[1:min(channel_size, length(image_files))]  
        Med_images = [load_images(file_path)[1] for file_path in needed_image_files]
        # Collect metadata for each image
        println("Collecting original metadata for channel '$channel_folder'.")
        metadata = [Dict(
            "file_path" => file,
            "data_split" => get_class_or_split_from_json(channel_path, dataset_splits),
            "class" => get_class_or_split_from_json(channel_path, class_mapping, class_names),
            "patient_uid_org" => img.patient_uid,
            "shape_org" => size(img.voxel_data),
            "spacing_org" => img.spacing,
            "origin_org" => img.origin,
            "direction_org" => img.direction,
            "type_org" => img.image_type
        ) for (file, img) in zip(needed_image_files, Med_images)]
        
        # Resample to the first image in the channel if required
        if resample_images_to_target && !isempty(Med_images)
            println("Resampling $channel_type files in channel '$channel_folder' to the first $channel_type in the channel.")
            reference_image = Med_images[1]
            Med_images = [resample_to_image(reference_image, img, interpolator) for img in Med_images]
        end

        # Handle cases where the number of images is less than or greater than the channel size
        if length(Med_images) < channel_size
            if batch_complete
                println("Channel '$channel_folder' has less than $channel_size $channel_type files. Padding with zeros.")
                append!(Med_images, [update_voxel_and_spatial_data(Med_images[1], zeros(size(Med_images[1].voxel_data)), Med_images[1].spacing,
                                      Med_images[1].origin, Med_images[1].direction)
                                      for _ in 1:(channel_size - length(Med_images))])
            else
                error_msg = "Error: Not enough $channel_type files in channel '$channel_folder'. Expected at least $channel_size, found $(length(Med_images))."
                println(error_msg)
                push!(error_messages, error_msg)
                continue
            end
        elseif length(image_files) > channel_size
            println("Channel '$channel_folder' has more than $channel_size $channel_type files. Trimming the excess.")
        end

        push!(channels_data, Med_images)  # Store channel data without concatenation
        push!(all_metadata, metadata)
    end

    # Error handling
    if !isempty(error_messages)
        for message in error_messages
            println(message)
        end
        error("There were errors with the input $channel_type channels. Please see the error messages above.")
    end

    # Ensure uniform spacing across the entire dataset
    if resample_images_spacing == "set"
        println("Resampling all $channel_type files to target spacing: $target_spacing")
        target_spacing = Tuple(Float32(s) for s in target_spacing)
        channels_data = [[resample_to_spacing(img, target_spacing, interpolator) for img in channel] for channel in channels_data]
    elseif resample_images_spacing == "avg"
        println("Calculating average spacing across all $channel_type files and resampling.")
        all_spacings = [img.spacing for channel in channels_data for img in channel]
        avg_spacing = Tuple(Float32(mean(s)) for s in zip(all_spacings...))
        println("Average spacing calculated: $avg_spacing")
        
        channels_data = [[resample_to_spacing(img, avg_spacing, interpolator) for img in channel] for channel in channels_data]
    elseif resample_images_spacing == "median"
        println("Calculating median spacing across all $channel_type files and resampling.")
        all_spacings = [img.spacing for channel in channels_data for img in channel]
        median_spacing = Tuple(Float32(median(s)) for s in all_spacings)
        println("Median spacing calculated: $median_spacing")
        channels_data = [[resample_to_spacing(img, median_spacing, interpolator) for img in channel] for channel in channels_data]
    elseif resample_images_spacing == false
        println("Skipping resampling of $channel_type files.")
        # No resampling will be applied, channels_data remains unchanged.
    end
  
    # Apply normalization and standardization if required
    if normalization && channel_type != "mask"
        println("Applying normalization to all $channel_type files.")
        channels_data = [[normalize_image(img) for img in channel] for channel in channels_data]
    end
    if standardization && channel_type != "mask"
        println("Applying standardization to all $channel_type files.")
        channels_data = [[standardize_image(img) for img in channel] for channel in channels_data]
    end

    # Adjust the dimensions of all images to the average or the stated dimension
    if resample_size == "avg"
        sizes = [size(img.voxel_data) for img in channels_data for img in img]  # Get sizes from all images
        avg_dim = map(mean, zip(sizes...))
        avg_dim = Tuple(Int(round(d)) for d in avg_dim)
        println("Resizing all $channel_type files to average dimension: $avg_dim")
        channels_data = [[crop_or_pad(img, avg_dim) for img in channel] for channel in channels_data]
    elseif resample_size != "avg" #TODO: change the wording, it is not readable
        target_dim = Tuple(resample_size)
        println("Resizing all $channel_type files to target dimension: $target_dim")
        channels_data = [[crop_or_pad(img, target_dim) for img in channel] for channel in channels_data]
    end

    # Ensure that all files have the same size
    expected_dim = size(channels_data[1][1].voxel_data)  # Assuming all images have been resized correctly
    for channel in channels_data
        for img in channel
            if size(img.voxel_data) != expected_dim
                error("$channel_type size mismatch. Expected size: $expected_dim, but got size: $(size(img.voxel_data))")
            end
        end
    end

    # Update metadata for each image
    for (i, channel) in enumerate(channels_data)
        meta_list = all_metadata[i]
        for (j, img) in enumerate(channel)
            meta = meta_list[j]
            meta["name"]=channel_names[i] * "_$j"
            meta["shape_final"] = size(img.voxel_data)
            meta["spacing_final"] = img.spacing
            meta["origin_final"] = img.origin
            meta["direction_final"] = img.direction
        end
    end

    channels_tensor = []
    for (i , channel) in enumerate(channels_data)
        # Concatenate all images in a channel into a single tensor along the 4th dimension
        println("Concatenating all $channel_type files into channels $i.")
        channel_tensor = cat([img.voxel_data for img in channel]..., dims=4)
        push!(channels_tensor, channel_tensor)
    end
    # Concatenating all channels into a single tensor with correct dimension handling
    println("Concatenating all $channel_type files into a single tensor.")
    final_tensor = cat(channels_tensor..., dims=5)
    println("Collecting all metadata")
    metadata = vcat(all_metadata...)
    final_tensor_size = size(final_tensor)
    println("Tensor formation complete. Returning the final $final_tensor_size tensor and metadata.")
    return final_tensor, metadata, channel_names
end