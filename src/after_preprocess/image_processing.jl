module image_processing
using Random

"""
Divide a 5D tensor into patches of given size, padding the image if necessary.
# Arguments
- `image`: A 5D tensor where the first dimension is the batch and the second is the channel.
- `patch_size`: A tuple specifying the size of the patch (excluding batch and channel dimensions).
# Returns
- A list of patches.
- The padding applied to each dimension.
"""
function divide_into_patches(image::Array{T, 5}, patch_size::Tuple{Int, Int, Int, Int}) where T
    # Calculate the required padding for each dimension
    pad_size = (0, 0, max(0, patch_size[1] - size(image, 3) % patch_size[1]),
                max(0, patch_size[2] - size(image, 4) % patch_size[2]),
                max(0, patch_size[3] - size(image, 5) % patch_size[3]))

    # Pad the image if necessary
    if any(pad_size .> 0)
        padded_image = pad(image, pad_size)
    else
        padded_image = image
    end

    patches = []
    for b in 1:size(padded_image, 1)
        for c in 1:size(padded_image, 2)
            for i in 1:patch_size[1]:size(padded_image, 3)
                for j in 1:patch_size[2]:size(padded_image, 4)
                    for k in 1:patch_size[3]:size(padded_image, 5)
                        patch = view(padded_image, b, c, i:min(i+patch_size[1]-1, size(padded_image, 3)),
                                     j:min(j+patch_size[2]-1, size(padded_image, 4)),
                                     k:min(k+patch_size[3]-1, size(padded_image, 5)))
                        push!(patches, patch)
                    end
                end
            end
        end
    end

    return patches, pad_size
end

"""
Recreate the original image from a list of patches, removing any padding applied.
# Arguments
- `patches`: A list of patches.
- `original_size`: The original size of the image before padding.
- `pad_size`: The padding applied to each dimension.
# Returns
- The original image with padding removed.
"""
function recreate_image_from_patches(patches::Vector{Array{T, 5}}, original_size::Tuple{Int, Int, Int, Int, Int}, pad_size::Tuple{Int, Int, Int, Int, Int}) where T
    padded_size = (original_size[1], original_size[2], original_size[3] + pad_size[3],
                   original_size[4] + pad_size[4], original_size[5] + pad_size[5])
    padded_image = zeros(T, padded_size...)

    patch_index = 1
    for b in 1:original_size[1]
        for c in 1:original_size[2]
            for i in 1:patch_size[1]:padded_size[3]
                for j in 1:patch_size[2]:padded_size[4]
                    for k in 1:patch_size[3]:padded_size[5]
                        patch = patches[patch_index]
                        padded_image[b, c, i:min(i+patch_size[1]-1, padded_size[3]),
                                     j:min(j+patch_size[2]-1, padded_size[4]),
                                     k:min(k+patch_size[3]-1, padded_size[5])] .= patch
                        patch_index += 1
                    end
                end
            end
        end
    end

    # Remove padding
    original_image = view(padded_image, 1:original_size[1], 1:original_size[2],
                          1:original_size[3], 1:original_size[4], 1:original_size[5])

    return original_image
end





"""
Perform largest connected components analysis on a 5D tensor.

# Arguments
- `tensor`: A 5D tensor where dimension 1 is batch and dimension 2 is channel.
- `config`: Configuration object containing the number of components to retain (`n_components`).

# Returns
- An integer array where all voxels from each connected component have a different integer associated with it, retaining only the largest `n_components`.
"""
function largest_connected_components(tensor, config::Configuration)
    batch_size, channels, dim1, dim2, dim3 = size(tensor)
    result = zeros(Int, size(tensor))

    for b in 1:batch_size
        for c in 1:channels
            component_labels = label_components(tensor[b, c, :, :, :])
            component_sizes = count_components(component_labels)
            sorted_components = sortperm(component_sizes, rev=true)
            largest_components = sorted_components[1:config.n_components]

            for i in 1:dim1
                for j in 1:dim2
                    for k in 1:dim3
                        label = component_labels[i, j, k]
                        if label in largest_components
                            result[b, c, i, j, k] = label
                        end
                    end
                end
            end
        end
    end

    return result
end

# Helper function to label connected components
function label_components(volume)
    labeled_volume, _ = connected_components(volume)
    return labeled_volume
end

# Helper function to count the size of each component
function count_components(labeled_volume)
    max_label = maximum(labeled_volume)
    component_sizes = zeros(Int, max_label)
    for label in 1:max_label
        component_sizes[label] = sum(labeled_volume .== label)
    end
    return component_sizes
end







end