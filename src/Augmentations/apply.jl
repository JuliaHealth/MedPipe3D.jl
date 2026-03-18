using JSON
using Random

"""
`apply_augmentations(images, config_path::String)`

A helper function for applying image augmentations as specified in a configuration file.

# Arguments
- `images`: A multidimensional (5-dim) array representing batches of images,
  of type => [HEIGHT, WIDTH, DEPTH, CHANNELS, BATCHES].
- `config_path`: The path to the JSON configuration file specifying augmentation
  types and their parameters.

# Returns
- The augmented image tensor after applying all specified transformations.

# Description
Loads augmentation settings from a configuration file and applies each augmentation
in the order specified to each image slice in the dataset.

Each augmentation entry carries its own `p_rand` field (probability of being applied).
If absent, the augmentation is always applied (p = 1.0).
"""
function apply_augmentations(images, config_path::String)
    config     = JSON.parsefile(config_path)
    aug_list   = config["augmentation"]["augmentations"]  # Vector of dicts, each with name/p_rand/params
    order      = config["augmentation"]["order"]          # Vector of augmentation names

    # Build a lookup: name → {p_rand, params}
    aug_lookup = Dict(a["name"] => a for a in aug_list)

    for b in axes(images, 5)        # batch
        for aug_name in order
            haskey(aug_lookup, aug_name) || continue
            entry  = aug_lookup[aug_name]
            p_rand = get(entry, "p_rand", 1.0)
            rand() < p_rand || continue

            params = entry["params"]
            for c in axes(images, 4)   # channel
                img_view = view(images, :, :, :, c, b)
                images[:, :, :, c, b] = apply_augmentation(img_view, aug_name, params)
            end
        end
    end

    return images
end

"""
`apply_augmentation(image, augmentation, params)`

Apply a single named augmentation to one image slice.

# Arguments
- `image`: A 3-D array slice `[H, W, D]`.
- `augmentation`: Name string matching one of the known augmentation keys.
- `params`: Dict of augmentation-specific parameters.

# Returns
- The augmented image array.
"""
function apply_augmentation(image, augmentation::String, params::Dict)
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
        # axes may be stored as a Vector{Int} (from JSON) or a String "(1,2,3)"
        axes_param = params["axes"]
        axes_tuple = if axes_param isa AbstractString
            eval(Meta.parse(axes_param))
        else
            Tuple(Int(a) for a in axes_param)
        end
        return augment_mirror(image, axes_tuple)

    elseif augmentation == "Scale transform"
        return augment_scaling(image, params["scale_factor"],
                               Symbol(params["interpolator_enum"]))

    elseif augmentation == "Gaussian blur transform"
        return augment_gaussian_blur(image,
                                     params["sigma"],
                                     params["shape"],
                                     params["kernel_size"],
                                     params["processing_unit"])

    elseif augmentation == "Simulate low-resolution transform"
        return augment_simulate_low_resolution(image,
                                               params["blur_sigma"],
                                               params["kernel_size"],
                                               params["downsample_scale"])

    elseif augmentation == "Elastic deformation transform"
        return elastic_deformation3d(image,
                                     params["strength"],
                                     Symbol(params["interpolator_enum"]))

    else
        error("Unknown augmentation: \"$augmentation\"")
    end
end