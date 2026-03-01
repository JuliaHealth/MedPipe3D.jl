using ITKIOWrapper

function batching_channeling!(tensor5D, input_images, number_of_batches, number_of_channels)
    expected_length = number_of_batches * number_of_channels
    if length(input_images) != expected_length
        error("Input image count mismatch: expected $(expected_length), got $(length(input_images)).")
    end

    reshaped_input_images = reshape(input_images, (number_of_batches, number_of_channels))
    
    for batch in 1:number_of_batches
        for channel in 1:number_of_channels
            image = reshaped_input_images[batch, channel]
            metadata = load_spatial_metadata(image)
            voxel_data = load_voxel_data(image, metadata)
            arr = voxel_data.dat
            arr_perm = permutedims(arr, (3, 1, 2)) # (depth, height, width)
            tensor5D[batch, channel, :, :, :] .= arr_perm
        end
    end
    return tensor5D
end