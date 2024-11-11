#region configuration 
function string_to_tuple(str::String)
    str_clean = replace(str, '(' => "", ')' => "")
    str_split = split(str_clean, ",")
    tuple_values = Tuple(parse(Float32, s) for s in str_split)
    return tuple_values
end
#endregion

#region batch creation
function crop_or_pad(img, target_size::Tuple)
    if isa(img, MedImage)
        current_size = size(img.voxel_data)
    else
        current_size = size(img)
    end
    #println("Current image size: ", current_size)
    #println("Target size: ", target_size)

    if current_size == target_size
        println("No cropping or padding needed.")
        return img
    end

    size_diff = map((a, b) -> a - b, current_size, target_size)
    #println("Size difference: ", size_diff)

    # Calculate crop and pad sizes for each dimension
    crop_beg = Tuple(max(0, floor(Int, size_diff[i] / 2)) + 1 for i in 1:length(size_diff))
    crop_size = Tuple(min(current_size[i], target_size[i]) for i in 1:length(size_diff))
    cropped_img = crop_mi(img, crop_beg, crop_size)
    # Calculate padding needed after cropping
    if isa(img, MedImage)
        #println("size:", size(cropped_img.voxel_data), "current_size:", current_size)
        after_crop_size = size(cropped_img.voxel_data)
    else
        #println("size:", size(cropped_img.voxel_data), "current_size:", current_size)
        after_crop_size = size(cropped_img)
    end
    
    pad_size_diff = map((a, b) -> b - a, after_crop_size, target_size)
    pad_beg = Tuple(max(0, floor(Int, pad_size_diff[i] / 2)) for i in 1:length(pad_size_diff))
    pad_end = Tuple(pad_size_diff[i] - pad_beg[i] for i in 1:length(pad_size_diff))

    # Apply padding if needed
    if any(x -> x != 0, pad_size_diff)
        #println("Padding required: begin ", pad_beg, ", end ", pad_end)
        padded_img = pad_mi(cropped_img, pad_beg, pad_end, 0)
        return padded_img
    else
        return cropped_img
    end
end

function normalize_image(img::MedImage)::MedImage
    voxel_data = img.voxel_data
    min_val = minimum(voxel_data)
    max_val = maximum(voxel_data)
    normalized_data = (voxel_data .- min_val) ./ (max_val - min_val + eps(Float32))
    return update_voxel_and_spatial_data(img, normalized_data, img.spacing, img.origin, img.direction)
end

function standardize_image(img::MedImage)::MedImage
    voxel_data = img.voxel_data
    mean_val = mean(voxel_data)
    std_val = std(voxel_data)
    standardized_data = (voxel_data .- mean_val) ./ (std_val + eps(Float32))
    return update_voxel_and_spatial_data(img, standardized_data, img.spacing, img.origin, img.direction)
end

function safe_write_meta(group::HDF5.Group, meta::Dict{String, Any})
    for (key, value) in meta
        if isa(value, Tuple) || isa(value, NTuple)
            safe_value = isa(value, NTuple) ? collect(value) : safe_value
            write_attribute(group, key, safe_value)
        elseif isa(value, Image_type)
            safe_value = isa(value, Image_type) ? string(value) : value
            write_attribute(group, key, string(value))
        elseif value === nothing
            continue
        else
            write_attribute(group, key, value)
        end
    end
end

function get_class_or_split_from_json(channel_path::String, json_path, class_names = nothing)
    # If the JSON path is false, return nothing
    if json_path == false
        return nothing
    else
        data = JSON.parsefile(json_path)
    end

    # Assign unique indices to each class name
    if class_names !== nothing
        indexed_class_names_dict = Dict(name => "$(i)_$name" for (i, name) in enumerate(class_names))
        for (org_class_name, paths) in data
            if any(occursin(p, channel_path) for p in paths)
                return indexed_class_names_dict[org_class_name]
            end
        end
    else
        for (key, paths) in data
            if any(occursin(p, channel_path) for p in paths)
                return key  
            end
        end
    end
    # Iterate over data and match paths with indexed class names

    return nothing
end
#endregion

#region visualization HDF5
function print_hdf5_contents(hdf5_path::String)
    function print_group(name::String, obj, indent::Int=0)
        indent_str = "  "^indent
        if isa(obj, HDF5.Group) || isa(obj, HDF5.File)
            println("$indent_str- Group: $name")
            # Print attributes of the group
            for attr_name in keys(attrs(obj))
                attr_value = read_attribute(obj, attr_name)
                println("$indent_str    Attribute: $attr_name = $attr_value")
            end
            # Iterate over members
            for member_name in keys(obj)
                member_obj = obj[member_name]
                print_group(member_name, member_obj, indent + 1)
            end
        elseif isa(obj, HDF5.Dataset)
            println("$indent_str- Dataset: $name")
            dataset_shape = size(obj)
            println("$indent_str    Shape: $dataset_shape")
            # Print attributes of the dataset
            for attr_name in keys(attrs(obj))
                attr_value = read_attribute(obj, attr_name)
                println("$indent_str    Attribute: $attr_name = $attr_value")
            end
        else
            println("$indent_str- Unknown object type: $name")
        end
    end

    h5open(hdf5_path, "r") do file
        println("Contents of HDF5 file '$hdf5_path':")
        print_group("/", file)
    end
end

#TODO: po zmianie strukury hdf5 prawdopodonie przestało działać
function load_images_from_hdf5(hdf5_path::String)
    image_batches = []
    image_batch_metadata = []
    mask_batches = []
    mask_batch_metadata = []

    h5open(hdf5_path, "r") do file
        for batch_name in keys(file)
            batch_group = file[batch_name]
            if haskey(batch_group, "images")
                images_data, image_meta = process_data_group(batch_group, "images", "images_metadata")
                push!(image_batches, images_data)
                push!(image_batch_metadata, image_meta)
            else
                println("No 'images' dataset found in '$batch_name'.")
            end
            if haskey(batch_group, "masks")
                masks_data, mask_meta = process_data_group(batch_group, "masks", "masks_metadata")
                push!(mask_batches, masks_data)
                push!(mask_batch_metadata, mask_meta)
            else
                println("No 'masks' dataset found in '$batch_name'.")
            end
        end
    end
    return image_batches, image_batch_metadata, mask_batches, mask_batch_metadata
end

function process_data_group(batch_group, data_key, metadata_key)
    data = read(batch_group[data_key])
    num_channels = size(data, 4)
    num_images_in_batch = size(data, 5)
    channel_data = []
    channel_meta = []

    if haskey(batch_group, metadata_key)
        meta_group = batch_group[metadata_key]
        for i in 1:num_images_in_batch
            images_in_channel = []
            meta_in_channel = []
            for j in 1:num_channels
                if data_key == "images"
                    meta_key = "image_$(j)"
                else
                    meta_key = "mask_$(j)"
                end
                meta = haskey(meta_group, meta_key) ? read_metadata(meta_group[meta_key]) : Dict()
                push!(meta_in_channel, meta)
                img_data = data[:, :, :, j, i]
                push!(images_in_channel, img_data)
            end
            push!(channel_data, images_in_channel)
            push!(channel_meta, meta_in_channel)
        end
    else
        println("No '$metadata_key' group found in '$batch_group'.")
    end

    return channel_data, channel_meta
end

function read_metadata(meta_group)
    meta = Dict()
    for attr_name in keys(attrs(meta_group))
        attr_value = read_attribute(meta_group, attr_name)
        meta[attr_name] = attr_value
    end
    return meta
end

function process_and_save_medimage(meta, data, output_folder, suffix)
    original_file_path = meta["file_path"]
    original_image = load_images(original_file_path)[1]  # Load the original MedImage

    updated_image = update_voxel_and_spatial_data(
        original_image, data, original_image.origin, original_image.spacing, original_image.direction
    )

    filename_without_ext, ext = splitext(basename(original_file_path))
    new_filename = filename_without_ext * suffix * ext
    output_file_path = joinpath(output_folder, new_filename)

    create_nii_from_medimage(updated_image, output_file_path)
    if suffix == "_mask_after"
        println("Saved updated mask to: $output_file_path")
    elseif suffix == "_image_after"
        println("Saved updated image to: $output_file_path")
    end
end

function convert_hdf5_to_medimages(hdf5_path::String, output_folder::String)
    image_batches, image_batch_metadata, mask_batches, mask_batch_metadata = load_images_from_hdf5(hdf5_path)

    # Process images
    for (batch_images, batch_meta) in zip(image_batches, image_batch_metadata)
        for (channel_images, channel_meta) in zip(batch_images, batch_meta)
            for (img_data, meta) in zip(channel_images, channel_meta)
                process_and_save_medimage(meta, img_data, output_folder, "_image_after")
            end
        end
    end
    # Process masks
    for (batch_masks, batch_meta) in zip(mask_batches, mask_batch_metadata)
        for (channel_masks, channel_meta) in zip(batch_masks, batch_meta)
            for (mask_data, meta) in zip(channel_masks, channel_meta)
                process_and_save_medimage(meta, mask_data, output_folder, "_mask_after")
            end
        end
    end
    
    println("All images and masks have been processed and saved.")
end
#endregion

#region model
function infer_model(tstate, model, data)
    println("Infering model")
    y_pred, st = Lux.apply(model, CuArray(data), tstate.parameters, tstate.states)
    return y_pred, st
end

function check_if_binary_and_report(tensor)
    # Initialize an empty array to store information about non-binary values and a set to track unique types
    non_binary_values = []
    unique_types = []

    # Check each element in the tensor
    for (index, value) in enumerate(tensor)
        push!(non_binary_values, (index, value))
        push!(unique_types, typeof(value))
    end
    println("non_binary_values: ", non_binary_values)
    println("unique_types: ", unique_types)

end

function is_binary_tensor(tensor)
    # Check if all elements in the tensor are either 0 or 1
    return all(x -> x == 0 || x == 1, tensor)
end
#endregion

