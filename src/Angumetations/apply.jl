function apply_augmentations(images, config_path::String)
    # Load the configuration
    config = JSON.parsefile(config_path)
    aug_config = config["augmentations"]
    order = config["order"]

    # Apply each augmentation in the specified order
    for b in 1:size(images, 5)  # Loop over each batch
        for c in 1:size(images, 4)  # Loop over each channel in the batch
            for aug in order  # Apply each augmentation in the specified order
                params = aug_config[aug]
                # Apply the augmentation to each slice individually
                images[:, :, :, c, b] = apply_augmentation(Array(images[:, :, :, c, b]), aug, params)
            end
        end
    end
    return images
end

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
        return augment_mirror(image, parse(Tuple{Int,Int,Int}, params["axes"]))
    elseif augmentation == "Scale transform"
        return augment_scaling(image, params["interpolator_enum"])
    elseif augmentation == "Gaussian blur transform"
        return augment_gaussian_blur(image, params["sigma"], params["kernel_size"], params["shape"])
    elseif augmentation == "Simulate low-resolution transform"
        return augment_simulate_low_resolution(image, params["blur_sigma"], params["kernel_size"], params["downsample_scale"])
    elseif augmentation == "Elastic deformation transform"
        return elastic_deformation3d(image, params["strength"], params["interpolator_enum"])
    end
end