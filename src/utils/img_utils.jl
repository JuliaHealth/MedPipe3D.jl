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

    cropped = if any(>(0), size_diff)
        # Use 0-based offsets (number of voxels to skip), defaulting to 0
        crop_beg = ntuple(i -> size_diff[i] > 0 ? floor(Int, size_diff[i] / 2) : 0, ndim)

        # Use target_size for oversized axes, keep current_size for the rest
        crop_size = ntuple(i -> size_diff[i] > 0 ? target_size[i] : current_size[i], ndim)

        crop_mi(img, crop_beg, crop_size, interpolator)
    else
        img
    end
    # Centre-pad if any axis is still undersized
    after_size = size(cropped.voxel_data)
    pad_diff   = ntuple(i -> target_size[i] - after_size[i], ndim)
    any(>(0), pad_diff) || return cropped

    pad_beg = ntuple(i -> max(0, floor(Int, pad_diff[i] / 2)), ndim)
    pad_end = ntuple(i -> pad_diff[i] - pad_beg[i], ndim)
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
    v      = img.voxel_data
    lo, hi = minimum(v), maximum(v)
    normed = (v .- lo) ./ (hi - lo + eps(Float32))
    return update_voxel_and_spatial_data(img, normed, img.origin, img.spacing, img.direction)
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
    return update_voxel_and_spatial_data(img, zs, img.origin, img.spacing, img.direction)
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
    origin::NTuple{3, Float64}             = (0.0, 0.0, 0.0),
    spacing::NTuple{3, Float64}            = (1.0, 1.0, 1.0),
    direction::NTuple{9, Float64} = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    image_type::MedImages.Image_type       = first(instances(MedImages.Image_type)),
    image_subtype::MedImages.Image_subtype = first(instances(MedImages.Image_subtype)),
    patient_id::String                     = "unknown",
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
"""
`rotate_5d_batch(image_5d, axis, angle)`

A local translation layer that safely passes 5D tensors into the external 
3D `rotate_mi` function by iterating over the batch and channel dimensions.
"""
function rotate_5d_batch(image_5d::AbstractArray{T, 5}, axis::Tuple, angle::Float64) where T
	# Fast path: skip rotation entirely if angle is 0
	if angle == 0.0
		return image_5d
	end

	# Convert the (1,0,0) tuple into the Int that the external library expects
	axis_int = findfirst(x -> x != 0, axis)
	if isnothing(axis_int)
		return image_5d
	end

	println("Applying 3D rotation (axis=$axis_int, angle=$angle) to 5D batch...")
	out = similar(image_5d)

	for b in axes(image_5d, 5)
		for c in axes(image_5d, 4)
			# 1. Slice out the 3D volume
			vol = view(image_5d,:,:,:,c,b)

			# 2. Convert to the external library's expected MedImage struct
			mi_temp = medimage_from_array(vol)

			# 3. Call the external function explicitly with all required arguments
			# (Assuming Nearest_neighbour_en is exported by the external library)
			mi_rotated = rotate_mi(mi_temp, axis_int, angle, Nearest_neighbour_en, true)

			# 4. Put the voxel data back into the 5D tensor
			out[:, :, :, c, b] = mi_rotated.voxel_data
		end
	end

	return out
end
