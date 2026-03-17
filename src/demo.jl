# =============================================================================
# MedPipe3D Demo Pipeline
# Dataset: Medical Segmentation Decathlon - Task02_Heart
# Label: Left Atrium (single binary label)
# =============================================================================

try
	using MedPipe3D
catch
	include(joinpath(@__DIR__, "MedPipe3D.jl"))
	using .MedPipe3D
end
import MedImages
using HDF5

# =============================================================================
# 0. HELPERS (LOCAL MedPipe3D + MedImages)
# =============================================================================

function get_or_create_group(parent, name::AbstractString)
	return haskey(parent, name) ? parent[name] : create_group(parent, name)
end

function to_channel_last(arr)
	return reshape(arr, size(arr)..., 1)
end

function load_first_image(path::AbstractString, modality::AbstractString)
	if isdefined(MedImages, :load_images)
		return MedImages.load_images(path)[1]
	elseif isdefined(MedImages, :load_image) && hasmethod(MedImages.load_image, Tuple{String, String})
		return MedImages.load_image(path, String(modality))
	elseif isdefined(MedImages, :load_image) && hasmethod(MedImages.load_image, Tuple{String})
		return MedImages.load_image(path)
	end
	error("MedImages does not provide load_images or load_image")
end

function strip_nii_ext(path::AbstractString)
	base = basename(path)
	if endswith(base, ".nii.gz")
		return replace(base, ".nii.gz" => "")
	elseif endswith(base, ".nii")
		return replace(base, ".nii" => "")
	end
	return splitext(base)[1]
end

function load_image_and_label(img_path::AbstractString, lbl_path::AbstractString, modality::AbstractString)
	img = load_first_image(img_path, modality)
	lbl = load_first_image(lbl_path, modality)

	image_data = to_channel_last(Float32.(img.voxel_data))
	label_data = to_channel_last(UInt32.(lbl.voxel_data .== 1)) # Left atrium is label 1

	image_meta = Dict(
		"file_path" => img_path,
		"spacing" => img.spacing,
		"origin" => img.origin,
		"direction" => img.direction,
		"image_type" => string(img.image_type),
		"patient_uid_org" => img.patient_uid,
		"shape_org" => size(img.voxel_data)
	)
	mask_meta = Dict(
		"file_path" => lbl_path,
		"spacing" => lbl.spacing,
		"origin" => lbl.origin,
		"direction" => lbl.direction,
		"image_type" => string(lbl.image_type),
		"patient_uid_org" => lbl.patient_uid,
		"shape_org" => size(lbl.voxel_data)
	)

	return image_data, label_data, image_meta, mask_meta
end

function write_patient_group(fid, group_name::AbstractString, image_data, label_data, image_meta, mask_meta)
	gr = get_or_create_group(fid, group_name)
	gr_images = get_or_create_group(gr, "images")
	gr_masks = get_or_create_group(gr, "masks")

	gr_images["data"] = image_data
	gr_masks["data"] = label_data

	img_meta_group = get_or_create_group(gr_images, "metadata")
	mask_meta_group = get_or_create_group(gr_masks, "metadata")

	img_meta_entry = get_or_create_group(img_meta_group, "0")
	mask_meta_entry = get_or_create_group(mask_meta_group, "0")

	MedPipe3D.safe_write_meta(img_meta_entry, image_meta)
	MedPipe3D.safe_write_meta(mask_meta_entry, mask_meta)
end

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

data_dir   = joinpath(homedir(), "Task02_Heart")
images_dir = joinpath(data_dir, "imagesTr")
labels_dir = joinpath(data_dir, "labelsTr")
pathToHDF5 = joinpath(homedir(), "heart_dataset.hdf5")
# MedImages (this version) expects "CT" or "PET" for load_image(path, type).
# Use the closest option for your dataset.
image_modality = "CT"

# =============================================================================
# 2. FIND & PAIR IMAGE/LABEL FILES
# =============================================================================

image_files = sort(filter(f -> !startswith(basename(f), ".") &&
	!startswith(basename(f), "._") &&
	(endswith(f, ".nii") || endswith(f, ".nii.gz")),
	readdir(images_dir; join = true)))
label_files = sort(filter(f -> !startswith(basename(f), ".") &&
	!startswith(basename(f), "._") &&
	(endswith(f, ".nii") || endswith(f, ".nii.gz")),
	readdir(labels_dir; join = true)))

# Pair them up by filename stem (la_003, la_004, ...)
image_map = Dict(strip_nii_ext(f) => f for f in image_files)
label_map = Dict(strip_nii_ext(f) => f for f in label_files)
common_keys = sort(collect(intersect(keys(image_map), keys(label_map))))
zipped = [(image_map[k], label_map[k]) for k in common_keys]

missing_images = setdiff(keys(label_map), keys(image_map))
missing_labels = setdiff(keys(image_map), keys(label_map))
if !isempty(missing_images) || !isempty(missing_labels)
	println("Warning: mismatched image/label pairs.")
	!isempty(missing_images) && println("  Missing images for: ", sort(collect(missing_images)))
	!isempty(missing_labels) && println("  Missing labels for: ", sort(collect(missing_labels)))
end

println("Found $(length(zipped)) patients")
println("First pair:")
println("  Image : ", zipped[1][1])
println("  Label : ", zipped[1][2])

# =============================================================================
# 3. LOAD & PREPROCESS ONE PATIENT (MedImages)
# =============================================================================

tupl = zipped[1]
image_data, label_data, image_meta, mask_meta = load_image_and_label(tupl[1], tupl[2], image_modality)

println("Image size after preprocessing : ", size(image_data))
println("Label unique values            : ", unique(label_data))

# =============================================================================
# 4. SAVE TO HDF5 (MedPipe3D-compatible layout)
# =============================================================================

h5open(pathToHDF5, "w") do fid
	for (i, (img_path, lbl_path)) in enumerate(zipped)
		groupName = string(i - 1)
		println("Processing patient $groupName : ", basename(img_path))

		image_data, label_data, image_meta, mask_meta = load_image_and_label(img_path, lbl_path, image_modality)
		write_patient_group(fid, groupName, image_data, label_data, image_meta, mask_meta)
	end
end

println("All $(length(zipped)) patients processed and saved to: ", pathToHDF5)
