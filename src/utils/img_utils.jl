using MedImages
using Statistics: mean, std

# ────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────

"""
    crop_or_pad(img, target_size; interpolator, pad_val) -> MedImage

Crop and/or zero-pad `img` so its voxel data matches `target_size` exactly.

- Cropping is centre-aligned: excess voxels are removed equally from both ends.
- Padding is centre-aligned: deficit voxels are added equally to both ends,
  with any odd remainder placed at the end.
- If `size(img.voxel_data) == target_size` the image is returned unchanged.
"""
function crop_or_pad(
    img::MedImage,
    target_size::Tuple;
    interpolator::Interpolator_enum = MedImages.Nearest_neighbour_en,
    pad_val = 0,
)::MedImage
    current_size = size(img.voxel_data)
    current_size == target_size && return img

    ndim      = length(target_size)
    size_diff = ntuple(i -> current_size[i] - target_size[i], ndim)

    # Centre-crop
    crop_beg  = ntuple(i -> max(0, floor(Int, size_diff[i] / 2)) + 1, ndim)
    crop_size = ntuple(i -> min(current_size[i], target_size[i]),      ndim)
    cropped   = crop_mi(img, crop_beg, crop_size, interpolator)

    # Centre-pad if the crop did not reach target_size
    after_size   = size(cropped.voxel_data)
    pad_diff     = ntuple(i -> target_size[i] - after_size[i], ndim)
    any(!=(0), pad_diff) || return cropped

    pad_beg = ntuple(i -> max(0, floor(Int, pad_diff[i] / 2)), ndim)
    pad_end = ntuple(i -> pad_diff[i] - pad_beg[i],            ndim)
    return pad_mi(cropped, pad_beg, pad_end, pad_val, interpolator)
end

# ────────────────────────────────────────────────────────────
# Intensity normalisation
# ────────────────────────────────────────────────────────────

"""
    normalize_image(img) -> MedImage

Min-max normalise voxel data to [0, 1].
A small ε is added to the denominator to avoid division by zero on flat images.
"""
function normalize_image(img::MedImage)::MedImage
    v       = img.voxel_data
    lo, hi  = minimum(v), maximum(v)
    normed  = (v .- lo) ./ (hi - lo + eps(Float32))
    return update_voxel_and_spatial_data(img, normed, img.spacing, img.origin, img.direction)
end

"""
    standardize_image(img) -> MedImage

Z-score standardise voxel data (zero mean, unit variance).
A small ε is added to the standard deviation to avoid division by zero.
"""
function standardize_image(img::MedImage)::MedImage
    v  = img.voxel_data
    μ  = mean(v)
    σ  = std(v)
    zs = (v .- μ) ./ (σ + eps(Float32))
    return update_voxel_and_spatial_data(img, zs, img.spacing, img.origin, img.direction)
end

# ────────────────────────────────────────────────────────────
# Construction helpers
# ────────────────────────────────────────────────────────────

"""
    medimage_from_array(arr; kwargs...) -> MedImage

Wrap a plain array in a `MedImage` with sensible defaults for all spatial metadata.
Useful when constructing synthetic images or converting model output tensors.

# Keyword arguments
| Argument         | Default                              | Description                     |
|:-----------------|:-------------------------------------|:--------------------------------|
| `origin`         | `(0.0, 0.0, 0.0)`                   | World-space origin              |
| `spacing`        | `(1.0, 1.0, 1.0)`                   | Voxel spacing (mm)              |
| `direction`      | identity matrix (row-major, 9-tuple) | Image orientation cosines       |
| `image_type`     | first `Image_type` enum value        |                                 |
| `image_subtype`  | first `Image_subtype` enum value     |                                 |
| `patient_id`     | `"unknown"`                          |                                 |
"""
function medimage_from_array(
    arr::AbstractArray;
    origin::NTuple{3,Float64}   = (0.0, 0.0, 0.0),
    spacing::NTuple{3,Float64}  = (1.0, 1.0, 1.0),
    direction::NTuple{9,Float64} = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    image_type::MedImages.Image_type       = first(instances(MedImages.Image_type)),
    image_subtype::MedImages.Image_subtype = first(instances(MedImages.Image_subtype)),
    patient_id::String = "unknown",
)::MedImage
    MedImage(
        voxel_data    = arr,
        origin        = origin,
        spacing       = spacing,
        direction     = direction,
        image_type    = image_type,
        image_subtype = image_subtype,
        patient_id    = patient_id,
    )
end