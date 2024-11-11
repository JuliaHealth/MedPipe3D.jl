#TODO: BIG DEVELOPMENTS in channel handling

"""
`evaluate_test_set_test(test_groups, h5, model, tstate, config)`

Executes the evaluation of a trained model on a specified test set, capturing and returning performance metrics.

# Arguments
- `test_groups`: A list of group paths within the HDF5 file that define the test dataset.
- `h5`: HDF5 file handle used to read test data.
- `model`: The trained machine learning model to evaluate.
- `tstate`: The state of the model, typically containing weights and potentially other parameters.
- `config`: Configuration dictionary that may influence how data is processed and how evaluation is performed.

# Returns
- A list of metrics collected during the evaluation of each test group, providing insights into model performance.

# Errors
- Raises errors if there are issues accessing the test data, or if there are configuration mismatches during data processing or evaluation.
"""
function evaluate_test_set_test(test_groups, h5, model, tstate, config)
    println("Evaluating test set...")
    all_test_metrics = []
    config["learning"]["patch_probabilistic_oversampling"] = false
    for test_group in test_groups
        test_data, test_label, attributes = fetch_and_preprocess_data([test_group], h5, config)
        results, test_metrics = evaluate_patches(test_data, test_label,  tstate, model, config)
        y_pred, metr = process_results(results, test_metrics, config)
        save_results_test(y_pred, attributes, config)
        push!(all_test_metrics, metr)
    end

    return all_test_metrics
end

"""
`evaluate_patches(test_data, test_label, tstate, model, config, axis, angle)`

Evaluates the model on rotated test data patches to assess robustness against geometric transformations. Returns a tuple of results and metrics.

# Arguments
- `test_data`: Test images.
- `test_label`: Corresponding labels.
- `tstate`: Model state with weights.
- `model`: Trained model.
- `config`: Settings for evaluation.
- `axis`: Rotation axis.
- `angle`: Rotation angle in degrees.
"""
function evaluate_patches(test_data, test_label, tstate, model, config, axis, angle)
    println("Evaluating patches...")
    results = []
    test_metrics = []
    tstates = [tstate]
    test_time_augs = []

    for i in config["learning"]["n_invertible"]
        data = rotate_mi(test_data, axis, angle)
        for tstate_curr in tstates
            patch_results = []
            patch_size = Tuple(config["learning"]["patch_size"])
            idx_and_patches, paded_data_size = divide_into_patches_test(test_data, patch_size)
            coordinates = [patch[1] for patch in idx_and_patches]
            patch_data = [patch[2] for patch in idx_and_patches]
            for patch in patch_data
                y_pred_patch, _ = infer_model(tstate_curr, model, patch)
                push!(patch_results, y_pred_patch)
            end
            idx_and_y_pred_patch = zip(coordinates, patch_results)
            y_pred = recreate_image_from_patches_test(idx_and_y_pred_patch, paded_data_size, patch_size, size(test_data))
            if config["learning"]["largest_connected_component"]
                y_pred = largest_connected_component(y_pred, config["learning"]["n_lcc"])
            end
            metr = evaluate_metric(y_pred, test_label, config["learning"]["metric"])
            push!(test_metrics, metr)
        end
    end
    return results, test_metrics
end

"""
`evaluate_patches(test_data, test_label, tstate, model, config, axis, angle)`

Evaluates the model on rotated test data patches to assess robustness against geometric transformations. Returns a tuple of results and metrics.

# Arguments
- `test_data`: Test images.
- `test_label`: Corresponding labels.
- `tstate`: Model state with weights.
- `model`: Trained model.
- `config`: Settings for evaluation.
- `axis`: Rotation axis.
- `angle`: Rotation angle in degrees.
"""
function divide_into_patches_test(image::AbstractArray{T, 5}, patch_size::Tuple{Int, Int, Int}) where T
    println("Dividing image into patches...")
    println("Size of the image: ", size(image)) 

    # Calculate the required padding for each dimension (W, H, D)
    pad_size = (
        (size(image, 1) % patch_size[1]) != 0 ? patch_size[1] - size(image, 1) % patch_size[1] : 0,
        (size(image, 2) % patch_size[2]) != 0 ? patch_size[2] - size(image, 2) % patch_size[2] : 0,
        (size(image, 3) % patch_size[3]) != 0 ? patch_size[3] - size(image, 3) % patch_size[3] : 0
    )

    # Pad the image if necessary
    padded_image = image
    if any(pad_size .> 0)
        padded_image = crop_or_pad(image, (size(image, 1) + pad_size[1], size(image, 2) + pad_size[2], size(image, 3) + pad_size[3]))
    end

    # Extract patches
    patches = []
    for x in 1:patch_size[1]:size(padded_image, 1)
        for y in 1:patch_size[2]:size(padded_image, 2)
            for z in 1:patch_size[3]:size(padded_image, 3)
                patch = view(
                    padded_image,
                    x:min(x+patch_size[1]-1, size(padded_image, 1)),
                    y:min(y+patch_size[2]-1, size(padded_image, 2)),
                    z:min(z+patch_size[3]-1, size(padded_image, 3)),
                    :,
                    :
                )
                push!(patches, [(x, y, z), patch])
            end
        end
    end
    println("Size of padded image: ", size(padded_image))
    return patches, size(padded_image)
end

"""
`recreate_image_from_patches_test(coords_with_patches, padded_size, patch_size, original_size)`

Reassembles a full image from its patches, adjusting for any initial padding, and returns the image cropped to its original size.

# Arguments
- `coords_with_patches`: List of (coordinates, patch) tuples.
- `padded_size`: Dimensions after padding.
- `patch_size`: Patch dimensions.
- `original_size`: Original image dimensions.
"""
function recreate_image_from_patches_test(
    coords_with_patches,
    padded_size,
    patch_size,
    original_size
)
    println("Recreating image from patches...")
    reconstructed_image = zeros(Float32, padded_size...)
    
    # Place patches back into their original positions
    for (coords, patch) in coords_with_patches
        x, y, z = coords
        reconstructed_image[
            x:x+patch_size[1]-1,
            y:y+patch_size[2]-1,
            z:z+patch_size[3]-1,
            :,
            :
        ] = patch
    end

    # Crop the reconstructed image to remove any padding
    final_image = reconstructed_image[
        1:original_size[1],
        1:original_size[2],
        1:original_size[3],
        :,
        :
    ]
    println("Size of the final image: ", size(final_image))
    return final_image
end


#TODO: rethink probably not needed in the new structure
function process_results_test(results, test_metrics, config)
    println("Processing results...")
    for i in 1:length(results)
        println("Processing result $i...")
        threshold = 0.5  # Adjust threshold as needed
        binary_result = results[i] .> threshold
        # Assuming we process one sample at a time
        # Extract the spatial dimensions and channel
        binary_result = binary_result[:, :, :, 1, 1]  # Extract spatial dimensions
        results[i] = largest_connected_components(binary_result)
    end
    y_pred = mean(results)
    return y_pred, metr
end