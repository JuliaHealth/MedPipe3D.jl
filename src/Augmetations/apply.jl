"""
`apply_augmentations(images, config_path::String)`

A helper function for applying image augmentations as specified in a configuration file.

# Arguments
- `images`: A multidimensional array representing batches of images.
- `config_path`: The path to the JSON configuration file specifying augmentation types and their parameters.

# Returns
- The augmented image tensor after applying all specified transformations.

# Description
Loads augmentation settings from a configuration file and applies each augmentation in the order specified to each image slice in the dataset.
Each augmentation is applied with probability p_rand from the configuration.
"""
function apply_augmentations(images, config_path::String)
    # Load the configuration
    config = JSON.parsefile(config_path)
    aug_config = config["augmentation"]["augmentations"]
    order = config["augmentation"]["order"]
    p_rand = get(config["augmentation"], "p_rand", 0.5)

    # Apply each augmentation in the specified order with probability p_rand
    for b in axes(images, 5)  # Loop over each batch
        for c in axes(images, 4)  # Loop over each channel in the batch
            for aug in order  # Apply each augmentation in the specified order
                # Apply augmentation with probability p_rand
                if rand() < p_rand
                    params = aug_config[aug]
                    # Apply the augmentation to each slice individually
                    images[:, :, :, c, b] = apply_augmentation(Array(images[:, :, :, c, b]), aug, params)
                end
            end
        end
    end
    return images
end

"""
`apply_augmentation(image, augmentation, params)`

A helper function for `apply_augmentations` that applies a specific augmentation to an image according to the provided parameters.

# Arguments
- `image`: A single image slice to which the augmentation will be applied.
- `augmentation`: The name of the augmentation to apply.
- `params`: A dictionary of parameters specific to the augmentation.

# Returns
- The augmented image.

# Description
Based on the augmentation name, this function adjusts the image's properties using the specified parameters.
"""
function apply_augmentation(image, augmentation, params)
    # Match the augmentation name and apply with parameters
    if augmentation == "Brightness transform"
        return augment_brightness(image, params["value"], params["mode"])
    elseif augmentation == "Contrast augmentation transform"
        return augment_contrast(image, params["factor"])
    elseif augmentation == "Gamma transform"
        return augment_gamma(image, params["gamma"])
    elseif augmentation == "Gaussian noise transform"
        return augment_gaussian_noise(image, params["variance"])
    elseif augmentation == "Rician noise transform"
        return augment_rician_noise(image, params["variance"])
    elseif augmentation == "Mirror transform"
        # Handle axes parameter - it might be a string or tuple
        axes_param = params["axes"]
        if axes_param isa String
            axes_tuple = eval(Meta.parse(axes_param))
        else
            axes_tuple = axes_param
        end
        return augment_mirror(image, axes_tuple)
    elseif augmentation == "Scale transform"
        return augment_scaling(image, params["scale_factor"], Symbol(params["interpolator_enum"]))
    elseif augmentation == "Gaussian blur transform"
        return augment_gaussian_blur(image, params["sigma"], params["shape"], params["kernel_size"], params["processing_unit"])
    elseif augmentation == "Simulate low-resolution transform"
        return augment_simulate_low_resolution(image, params["blur_sigma"], params["kernel_size"], params["downsample_scale"])
    elseif augmentation == "Elastic deformation transform"
        return elastic_deformation3d(image, params["strength"], Symbol(params["interpolator_enum"]))
    else
        error("Unknown augmentation: $augmentation")
    end
end