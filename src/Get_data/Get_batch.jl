"""
`fetch_and_preprocess_data(group_paths::Vector{String}, h5::HDF5.File, config::Dict{String, Any})`

Fetches and preprocesses medical image data based on specified configurations, optionally using probabilistic oversampling and hardware acceleration.

# Arguments
- `group_paths`: A vector of paths specifying where the data groups are located within the HDF5 file.
- `h5`: An open HDF5 file object from which data is to be fetched.
- `config`: A dictionary containing configuration settings that dictate how data should be fetched and processed.

# Returns
- `Tuple`: A tuple containing arrays of images, labels, and a vector of unique class labels extracted from the dataset.

# Description
This function retrieves batches of images and their corresponding labels from an HDF5 file, applying probabilistic oversampling if specified in the configuration.
It also handles the transfer of data to a GPU if indicated by the configuration settings.

# Errors
- Raises an error if there are issues accessing data from the HDF5 file.
- Raises an error if configuration keys are missing or incorrectly formatted.
"""

function fetch_and_preprocess_data(group_paths::Vector, h5::HDF5.File, config::Dict)

    if config["learning"]["patch_probabilistic_oversampling"]
        println("Fetching and preprocessing data with probabilistic oversampling.")
        images, labels, class_labels = get_patch_batch_with_classes(group_paths, h5, config)
    else
        println("Fetching and preprocessing data.")
        images, labels, class_labels = get_batch_with_classes(group_paths, h5, config)
    end
    if config["augmentation"]["processing_unit"] == "GPU"
        images = CuArray(images)  # Move images to GPU
        labels = CuArray(labels)
    end

    # Extract unique classes from the metadata
    unique_classes = unique(class_labels)
    return images, labels, unique_classes
end


"""
`get_batch_with_classes(group_paths::Vector{String}, h5::HDF5.File, config::Dict{String, Any})`

Fetches batches of images and labels from specified groups within an HDF5 file, applying class-specific labeling based on configuration settings.

# Arguments
- `group_paths`: A vector of paths specifying locations within the HDF5 file from which to fetch data.
- `h5`: An open HDF5 file object used for data retrieval.
- `config`: A configuration dictionary that influences how data is fetched and labeled.

# Returns
- `Tuple`: Returns a tuple consisting of a tensor of images, a tensor of labels, and a vector of class labels.

# Description
It supports class-specific labeling by modifying the label data based on class indices derived from a JSON configuration.


# Errors
- Raises an error if there are issues accessing the specified paths within the HDF5 file.
- Raises an error if class indices are incorrectly formatted or absent when required.
"""
#TODO: add logs so you know if multi class or binary
function get_batch_with_classes(group_paths, h5::HDF5.File, config::Dict)
    images = []
    labels = []
    class_labels = []

    println("Processing groups to one batch: $group_paths")
    for path in group_paths
        image_data = read(h5[path * "/images/data"])
        label_data = read(h5[path * "/masks/data"])
        push!(images, image_data)

        if config["data"]["has_mask"]
        # Apply class-based labeling if class JSON path is provided
            if config["learning"]["class_JSON_path"] != false
                class_idx_name = get_class_labels([path], h5, config)
                class_idx, class_name= split(only(keys(class_idx_name)), "_")  # Find matching key
                class_idx = parse(Int, class_idx)
                label_data .= label_data .* class_idx # Multiply label data by class index
                println("Class index: $class_idx, Class name: $class_name")
            end
            push!(labels, label_data)
        end
    end
    if config["learning"]["class_JSON_path"] != false
        class_labels_dict = get_class_labels(group_paths, h5, config)
        push!(class_labels, class_labels_dict)
    else
        class_labels = [1]
    end
    images_tensor = cat(images..., dims=5)
    labels_tensor = cat(labels..., dims=5)
    return images_tensor, labels_tensor, class_labels
end



"""
`get_patch_batch_with_classes(group_paths::Vector{String}, h5::HDF5.File, config::Dict{String, Any})`

Fetches and processes patches of images and labels from specified groups within an HDF5 file, applying class-specific labeling based on configuration settings, particularly suited for training on patch-based input.

# Arguments
- `group_paths`: A vector of paths specifying locations within the HDF5 file from which to fetch data.
- `h5`: An open HDF5 file object used for data retrieval.
- `config`: A configuration dictionary detailing how patches should be extracted and processed.

# Returns
- `Tuple`: Returns a tuple consisting of a tensor of image patches, a tensor of label patches, and a vector of class labels.

# Description
It supports class-specific labeling by modifying the label data based on class indices derived from a JSON configuration.

# Errors
- Raises an error if there are issues accessing the specified paths within the HDF5 file.
- Raises an error if class indices are incorrectly formatted or absent when required.
"""

function get_patch_batch_with_classes(group_paths::Vector, h5::HDF5.File, config::Dict)
    images, labels, class_labels = [], [], []
    
    patch_size = Tuple(config["learning"]["patch_size"])
    println("Processing groups to one batch: $group_paths")

    for path in group_paths
        image_data = read(h5[path * "/images/data"])
        label_data = read(h5[path * "/masks/data"])
        if config["learning"]["class_JSON_path"] != false
            class_idx_name = get_class_labels([path], h5, config)
            class_idx, class_name= split(only(keys(class_idx_name)), "_")  # Find matching key
            class_idx = parse(Int, class_idx)
            label_data .= label_data .* class_idx
        end
        
        channel_images, channel_labels = [], []
        println("Cutting patches in channel: ", path)

        for channel in 1:size(image_data, 4)
            image_slize = copy(image_data[:, :, :, channel])
            label_slice = copy(label_data[:, :, :, channel])
            image_patch, label_patch = extract_patch(image_slize, label_slice, patch_size, config)

            push!(channel_images, copy(image_patch))
            push!(channel_labels, copy(label_patch))
        end

        if config["learning"]["class_JSON_path"] != false
            push!(class_labels, copy(class_idx))
        end
        channel_image_tensor = cat(channel_images..., dims=4)
        channel_label_tensor = cat(channel_labels..., dims=4)
        push!(images, copy(channel_image_tensor))
        push!(labels, copy(channel_label_tensor))
    end
    
    # Default to class 1 if no class labels are found
    isempty(class_labels) ? class_labels = [1] : nothing

    # Combine patches into tensors and log class info
    images_tensor = cat(images..., dims=5)
    labels_tensor = cat(labels..., dims=5)
    return images_tensor, labels_tensor, class_labels
end




"""
`get_class_labels(patient_groups::Vector, h5::HDF5.File, config::Dict)`

A helper function for `get_batch_with_classes` and `get_patch_batch_with_classes`, responsible for extracting class labels and organizing groups by class based on metadata stored in an HDF5 file.

# Description
This function reads the class information from metadata and groups the data accordingly.
"""
function get_class_labels(patient_groups::Vector, h5::HDF5.File, config::Dict)
    # Extract class labels and organize groups by class
    class_groups = Dict{String, Vector{String}}()
    for patient_group_name in patient_groups
        # Read class from metadata
        metadata_images_path = patient_group_name * "/" * "images" * "/" * "metadata"
        metadata_masks_path = patient_group_name * "/" * "masks" * "/" * "metadata"

        class_label = ""
        found = false

        if group_exists(h5, metadata_images_path)
            metadata_group = h5[metadata_images_path]
            first_meta_subgroup_name = first(keys(metadata_group))
            meta_subgroup = metadata_group[first_meta_subgroup_name]
            if config["learning"]["class_JSON_path"] != false
                class_label = safe_read_attribute(meta_subgroup, "class")
            else
                class_label = "class1"
            end
            found = true
        elseif group_exists(h5, metadata_masks_path)
            metadata_group = h5[metadata_masks_path]
            first_meta_subgroup_name = first(keys(metadata_group))
            meta_subgroup = metadata_group[first_meta_subgroup_name]
            if config["learning"]["class_JSON_path"] != false
                class_label = safe_read_attribute(meta_subgroup, "class")
            else
                class_label = "class1"
            end
            found = true
        else
            println("No metadata found for $patient_group_name")
        end

        if found
            if haskey(class_groups, class_label)
                push!(class_groups[class_label], patient_group_name)
            else
                class_groups[class_label] = [patient_group_name]
            end
        end
    end
    return class_groups
end


"""
`get_class_labels(patient_groups::Vector, h5::HDF5.File, config::Dict)`

A helper function for `get_batch_with_classes` and `get_patch_batch_with_classes`, responsible for extracting class labels and organizing groups by class based on metadata stored in an HDF5 file.

# Description
This function reads the class information from metadata and groups the data accordingly.
"""
function get_class_mapping(patient_groups, h5, config)
    class_mapping = Dict{String, Vector{String}}()  # Define dictionary with String keys and Vector values
    class_counter = Dict{String, Int}()  # Track index for each unique class name

    for group in patient_groups
        # Fetch the class label dictionary and extract the class name
        class_dict = get_class_labels([group], h5, config)
        class_name = only(keys(class_dict))  # Extract the single class name key

        # Verify that class_name is a String
        if !isa(class_name, String)
            error("Expected class_name to be a String, got $(typeof(class_name))")
        end

        # Get or assign a unique index to the class name
        if !haskey(class_counter, class_name)
            class_counter[class_name] = length(class_counter) + 1  # Assign a new index
        end
        unique_class_key = "$(class_counter[class_name])_$class_name"  # Create unique key with index

        # Initialize or append to the class list in the dictionary
        if haskey(class_mapping, unique_class_key)
            push!(class_mapping[unique_class_key], group)
        else
            class_mapping[unique_class_key] = [group]
        end
    end

    return class_mapping
end


"""
`extract_patch(image, label, patch_size, config)`

A helper function for `get_patch_batch_with_classes`, used to extract patches from images and labels based on a probability defined in the configuration.

# Description
This function decides between extracting a patch centered on nonzero label values or a random patch, depending on a randomly generated number and a threshold probability from the configuration.
"""
function extract_patch(image, label, patch_size, config)
    # Fetch the oversampling probability from the config
    println("Extracting patch.")
    oversampling_probability = config["learning"]["oversampling_probability"]
    # Generate a random number to decide which patch extraction method to use
    random_choice = rand()

    if random_choice <= oversampling_probability
        return extract_nonzero_patch(image, label, patch_size)
    else

        return get_random_patch(image, label, patch_size)
    end
end


"""
`extract_nonzero_patch(image, label, patch_size)`

A helper function for `extract_patch`, aimed at extracting image patches centered around nonzero label values.

# Description
If nonzero label values are present, this function selects one at random to center the patch around; otherwise, it defaults to extracting a random patch.
"""
function extract_nonzero_patch(image, label, patch_size)
    println("Extracting a patch centered around a non-zero label value.")
    indices = findall(x -> x != 0, label)
    if isempty(indices)
        # Fallback to random patch if no non-zero points are found
        return get_random_patch(image, label, patch_size)
    else
        # Choose a random non-zero index to center the patch around
        center = indices[rand(1:length(indices))]
        return get_centered_patch(image, label, center, patch_size)
    end
end



"""
`get_centered_patch(image, label, center, patch_size)`

A helper function for `extract_nonzero_patch`, used to extract a patch from an image and label centered around a specific index.

# Description
Calculates the necessary padding and extracts the patch ensuring it is centered on the chosen index, adjusting dimensions as required.
"""
function get_centered_patch(image, label, center, patch_size)
    center_coords = Tuple(center)
    half_patch = patch_size .÷ 2
    start_indices = center_coords .- half_patch
    end_indices = start_indices .+ patch_size .- 1

    # Calculate padding needed
    pad_beg = (
        max(1 - start_indices[1], 0),
        max(1 - start_indices[2], 0),
        max(1 - start_indices[3], 0)
    )
    pad_end = (
        max(end_indices[1] - size(image, 1), 0),
        max(end_indices[2] - size(image, 2), 0),
        max(end_indices[3] - size(image, 3), 0)
    )

    # Adjust start_indices and end_indices after padding
    start_indices_adj = start_indices .+ pad_beg
    end_indices_adj = end_indices .+ pad_beg

    # Convert padding values to integers
    pad_beg = Tuple(round.(Int, pad_beg))
    pad_end = Tuple(round.(Int, pad_end))

    # Pad the image and label using pad_mi
    image_padded = pad_mi(image, pad_beg, pad_end, 0)
    label_padded = pad_mi(label, pad_beg, pad_end, 0)

    # Extract the patch
    image_patch = image_padded[
        start_indices_adj[1]:end_indices_adj[1],
        start_indices_adj[2]:end_indices_adj[2],
        start_indices_adj[3]:end_indices_adj[3]
    ]
    label_patch = label_padded[
        start_indices_adj[1]:end_indices_adj[1],
        start_indices_adj[2]:end_indices_adj[2],
        start_indices_adj[3]:end_indices_adj[3]
    ]

    return image_patch, label_patch
end


"""
`get_random_patch(image, label, patch_size)`

A helper function for `extract_patch`, which extracts a random patch from an image and label.

# Description
This function randomly selects a start index within the image and extracts a patch of specified size, adjusting the image dimensions if necessary to fit the patch size.
"""
function get_random_patch(image, label, patch_size)
    println("Extracting a random patch.")
    # Check if the patch size is greater than the image dimensions
    if any(patch_size .> size(image))
        # Calculate the needed size to fit the patch
        needed_size = map(max, size(image), patch_size)
        # Use crop_or_pad to ensure the image and label are at least as large as needed_size
        image = crop_or_pad(image, needed_size)
        label = crop_or_pad(label, needed_size)
    end

    # Calculate random start indices within the new allowable range
    start_x = rand(1:size(image, 1) - patch_size[1] + 1)
    start_y = rand(1:size(image, 2) - patch_size[2] + 1)
    start_z = rand(1:size(image, 3) - patch_size[3] + 1)
    start_indices = [start_x, start_y, start_z]
    end_indices = start_indices .+ patch_size .- 1

    # Extract the patch directly when within bounds
    image_patch = image[start_indices[1]:end_indices[1], start_indices[2]:end_indices[2], start_indices[3]:end_indices[3]]
    label_patch = label[start_indices[1]:end_indices[1], start_indices[2]:end_indices[2], start_indices[3]:end_indices[3]]

    return image_patch, label_patch
end