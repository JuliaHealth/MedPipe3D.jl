module get_batch

export get_patch, pad, get_random_patch, get_nonzero_patch, retrieve_image, maybe_extract_patch, stack_images, get_batch

using HDF5
using manage_indicies
using Random, Statistics

"""
Get a patch from a 4D image with probabilistic oversampling based on the label.
# Arguments
- `image`: A 4D array where the first dimension is the channel.
- `label`: A 4D array with the same shape as the image.
- `patch_size`: A tuple specifying the size of the patch.
- `p`: The probability for Bernoulli distribution.
# Returns
- A patch from the image.
"""
function get_patch(image::Array{T, 4}, label::Array{T, 4}, patch_size::Tuple{Int, Int, Int, Int}, p::Float64) where T
    # Check if patch size fits in the image
    for i in 1:4
        if patch_size[i] > size(image, i)
            error("Patch size does not fit in the image")
        end
    end

    # Evaluate Bernoulli with probability p
    if rand(Bernoulli(p)) == 0
        # Return a random part of the image
        return get_random_patch(image, patch_size)
    else
        # Return a part of the image that has at least one non-zero voxel in the label
        return get_nonzero_patch(image, label, patch_size)
    end
end

"""
Pad the image with zeros if the patch coordinates are outside the image boundaries.
# Arguments
- `image`: A 4D array where the first dimension is the channel.
- `pad_size`: A tuple specifying the padding size for each dimension.
# Returns
- A padded image.
"""
function pad(image::Array{T, 4}, pad_size::Tuple{Int, Int, Int, Int}) where T
    padded_image = zeros(T, size(image, 1) + pad_size[1],
                             size(image, 2) + pad_size[2],
                             size(image, 3) + pad_size[3],
                             size(image, 4) + pad_size[4])
    padded_image[1:size(image, 1), 1:size(image, 2), 1:size(image, 3), 1:size(image, 4)] .= image
    return padded_image
end

"""
Get a random patch from the image, padding the image if necessary.
# Arguments
- `image`: A 4D array where the first dimension is the channel.
- `patch_size`: A tuple specifying the size of the patch.
# Returns
- A random patch from the image.
"""
function get_random_patch(image::Array{T, 4}, patch_size::Tuple{Int, Int, Int, Int}) where T
    # Calculate the required padding for each dimension
    pad_size = (max(0, patch_size[1] - size(image, 1)),
                max(0, patch_size[2] - size(image, 2)),
                max(0, patch_size[3] - size(image, 3)),
                max(0, patch_size[4] - size(image, 4)))

    # Pad the image if necessary
    if any(pad_size .> 0)
        image = pad(image, pad_size)
    end

    # Calculate start indices ensuring they are within bounds
    start_indices = [rand(1:size(image, i) - patch_size[i] + 1) for i in 1:4]

    return view(image, start_indices[1]:start_indices[1] + patch_size[1] - 1,
                       start_indices[2]:start_indices[2] + patch_size[2] - 1,
                       start_indices[3]:start_indices[3] + patch_size[3] - 1,
                       start_indices[4]:start_indices[4] + patch_size[4] - 1)
end

"""
Get a patch from the image that has at least one non-zero voxel in the label, padding the image if necessary.
# Arguments
- `image`: A 4D array where the first dimension is the channel.
- `label`: A 4D array with the same shape as the image.
- `patch_size`: A tuple specifying the size of the patch.
# Returns
- A patch from the image that has at least one non-zero voxel in the label.
"""
function get_nonzero_patch(image::Array{T, 4}, label::Array{T, 4}, patch_size::Tuple{Int, Int, Int, Int}) where T
    # Find coordinates of all non-zero entries in the label
    nonzero_coords = findall(x -> x != 0, label)
    if isempty(nonzero_coords)
        return get_random_patch(image, patch_size)
    end

    # Randomly select one coordinate
    selected_coord = nonzero_coords[rand(1:end)]

    # Calculate the start indices for the patch
    start_indices = [max(1, selected_coord[i] - div(patch_size[i], 2)) for i in 1:4]
    end_indices = [min(size(image, i), start_indices[i] + patch_size[i] - 1) for i in 1:4]

    # Calculate the required padding for each dimension
    pad_size = (max(0, end_indices[1] - size(image, 1)),
                max(0, end_indices[2] - size(image, 2)),
                max(0, end_indices[3] - size(image, 3)),
                max(0, end_indices[4] - size(image, 4)))

    # Pad the image if necessary
    if any(pad_size .> 0)
        image = pad(image, pad_size)
        label = pad(label, pad_size)
    end

    # Adjust start indices if the patch exceeds the image boundaries
    for i in 1:4
        if end_indices[i] - start_indices[i] + 1 < patch_size[i]
            start_indices[i] = end_indices[i] - patch_size[i] + 1
        end
    end

    return view(image, start_indices[1]:start_indices[1] + patch_size[1] - 1,
                       start_indices[2]:start_indices[2] + patch_size[2] - 1,
                       start_indices[3]:start_indices[3] + patch_size[3] - 1,
                       start_indices[4]:start_indices[4] + patch_size[4] - 1)
end


"""
Retrieve a 4D image from HDF5 dataset using the given index.
# Arguments
- `hdf5_ref`: Reference to the HDF5 database.
- `index`: Index pointing to the HDF5 dataset.
# Returns
- A 4D image.
"""
function retrieve_image(hdf5_ref, index)
    return h5read(hdf5_ref, index)
end


"""
Retrieve a 4D label from HDF5 dataset using the given index.
# Arguments
- `hdf5_ref`: Reference to the HDF5 database.
- `index`: Index pointing to the HDF5 dataset.
# Returns
- A 4D image.
"""
function retrieve_label(hdf5_ref, index)
    return h5read(hdf5_ref, index)
end

"""
Extract a patch from the image if probabilistic oversampling is enabled.
# Arguments
- `image`: A 4D image.
- `label`: A 4D label with the same shape as the image.
- `config`: Configuration struct containing oversampling settings.
# Returns
- A patch from the image or the original image.
"""
function maybe_extract_patch(image, label, config)
    if config.use_probabilistic_oversampling
        return get_patch(image, label, config.patch_size, config.p)
    else
        return image
    end
end

"""
Stack a list of 4D images into a 5D tensor.
# Arguments
- `images`: A list of 4D images.
# Returns
- A 5D tensor.
"""
function stack_images(images)
    return cat(images..., dims=1)
end

"""
Get a batch of images from the HDF5 database.
# Arguments
- `indices`: List of indices pointing to the HDF5 datasets.
- `hdf5_ref`: Reference to the HDF5 database.
- `config`: Configuration struct containing oversampling settings.
- `is_label`: Boolean indicating whether to load labels instead of images.
# Returns
- A 5D tensor containing the batch of images.
"""
function get_batch(indices, hdf5_ref, config, is_label::Bool)
    images = []
    for index in indices
        if is_label
            image = retrieve_label(hdf5_ref, index)
        else
            image = retrieve_image(hdf5_ref, index)
        end
        label = retrieve_image(hdf5_ref, index)  # Assuming label is stored in the same index
        image = maybe_extract_patch(image, label, config)
        push!(images, image)
    end
    return stack_images(images)
end





"""
Fetches and preprocesses test data and labels for a given index based on the provided configuration.

# Arguments
- `indices_list`: The list of indices of the test data to fetch.
- `config::Configuration`: Configuration object that specifies whether labels are present and the device to use (CPU or CUDA).

# Returns
- `data`: The data.
- `label`: The label (if `label_present` is true in the configuration).

# Example
```julia
config = Configuration(label_present=true, device="CUDA")
test_data, test_label = fetch_and_preprocess_data([1, 2, 3], config)
"""
function fetch_and_preprocess_data(indices_list, hdf5_ref, config::Configuration) 
    
    data = get_batch(indices_list, hdf5_ref, config, false)
    label = nothing
    
     if config.label_present
        label = get_batch(indices_list, hdf5_ref, config, true)
    end
    
    if config.device == "CUDA"
        data = CUDA.array(test_data)
        if config.label_present
            label = CUDA.array(test_label)
        end
    end
    attributes=map(in->manage_indicies.read_attributes(hdf5_ref,in),indices_list)
    return data, label
end


end # module


