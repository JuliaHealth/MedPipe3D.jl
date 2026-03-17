# Channel convention (consistent with batch_loader.jl, validation.jl, model_utils.jl):
#
#   dim 1-3 → spatial (W, H, D)
#   dim 4   → channels
#               • for model INPUT:  input modalities (e.g. T1, T2, FLAIR)
#               • for model OUTPUT: class logits / predictions per class
#   dim 5   → batch (one element per patient group)
#
# Both input modalities and output class logits live on dim 4 because the 3-D
# segmentation model maps C_in input channels → C_out class channels over the
# same spatial volume.  This is the same convention used by:
#   - get_patch_batch_with_classes  (cat on dims=4 for channels, dims=5 for batch)
#   - evaluate_metric               (size(y_pred, 4) == n_classes)
#   - infer_model                   (passes 5-D tensor directly to Lux.apply)
#
# #TODO: BIG DEVELOPMENTS — the dual meaning of dim 4 (modalities in / classes out)
# is an acknowledged design tension.  Once the architecture is finalised (e.g. a
# dedicated channel-remapping layer or explicit reshape between encoder/decoder),
# this comment and the functions below should be revisited.

"""
`save_results_test(y_pred, attributes, config, h5)`
"""
function save_results_test(y_pred, attributes, config, h5::HDF5.File)
	println("Saving test results...")

	# Define the group name where predictions will live in the HDF5 file
	group_name = "test_predictions"

	process_and_save_medimage_test(attributes, y_pred, h5, group_name)
end
"""
`evaluate_test_set_test(test_groups, h5, model, tstate, config)`

Executes the evaluation of a trained model on a specified test set, 
capturing and returning performance metrics. Saves predictions directly to HDF5.
"""
function evaluate_test_set_test(test_groups, h5, model, tstate, config)
	println("Evaluating test set...")
	all_test_metrics = []
	all_test_std     = []

	# Disable oversampling during test evaluation
	config["learning"]["patch_probabilistic_oversampling"] = false

	for test_group in test_groups
		# 1. Fetch data and ground truth labels
		test_data, test_label, attributes = fetch_and_preprocess_data([test_group], h5, config)

		# 2. Run patch-based inference (handles spatial slicing and reconstruction)
		# If invertible TTA is enabled, this returns both mean and std volumes.
		results_mean, results_std, test_metrics = evaluate_patches(test_data, test_label, tstate, model, config)

		# 3. Process ensemble results WITH the ground truth labels to calculate final metric
		y_pred, metr = process_results_test(results_mean, test_metrics, test_label, config)

		# Also aggregate std across ensemble members (same aggregation as mean: element-wise average)
		# This yields a stable per-voxel uncertainty estimate even when multiple ensemble members exist.
		y_std = isempty(results_std) ? nothing : mean(results_std)

		# 4. Save predictions directly to the open HDF5 file
		save_results_test(y_pred, attributes, config, h5)

		push!(all_test_metrics, metr)
		push!(all_test_std, y_std)
	end

	return all_test_metrics, all_test_std
end
"""
`evaluate_patches(test_data, test_label, tstate, model, config, axis, angle)`

Evaluates the model on (optionally rotated) test data by dividing it into
non-overlapping spatial patches, running inference per patch, reconstructing the
full prediction volume, and computing the configured evaluation metric.

Channel handling:
- test_data arrives as [W, H, D, C_in, B] — dim 4 holds input modalities.
  All modality channels are passed to the model together as a single patch tensor.
- The model outputs [W, H, D, C_out, B] — dim 4 becomes class logits.
  This is passed intact through reconstruction and into evaluate_metric,
  which iterates over size(y_pred, 4) as n_classes.
"""
function evaluate_patches(test_data, test_label, tstate, model, config, axis = (1, 0, 0), angle = 0.0)
	println("Evaluating patches...")

	# 1. FORCE TO HOST CPU: Slicing, padding, and MedImage ops must happen on the CPU.
	# infer_model will automatically push individual patches back to the GPU later.
	test_data_cpu  = Array(test_data)
	test_label_cpu = Array(test_label)

	# For backward compatibility we still return a `results` vector, but each entry
	# is now the *mean* prediction volume from one evaluation pass.
	# When invertible TTA is enabled, we also return a parallel std volume.
	results_mean = []
	results_std  = []
	test_metrics = []
	tstates      = [tstate]

	# Optional invertible test-time augmentations (TTA)
	# - When enabled, `infer_model_tta` runs inference n times per patch and returns
	#   mean/std masks after inverse transforms.
	# - When disabled, it falls back to a single pass and std=0.
	for _ in 1:max(1, Int(get(config["learning"], "n_invertible", 1)))
		# Use the CPU version of the data here
		# (Note: If you renamed this to rotate_5d_batch earlier, use that name here!)
		data = rotate_5d_batch(test_data_cpu, axis, angle)

		for tstate_curr in tstates
			_patch_cfg = get(config["learning"], "patch", Dict())
			patch_size = Tuple(
				isa(_patch_cfg, Dict) && haskey(_patch_cfg, "size") ?
				_patch_cfg["size"] :
				config["learning"]["patch_size"],
			)

			idx_and_patches, padded_data_size = divide_into_patches_test(data, patch_size)
			coordinates = [p[1] for p in idx_and_patches]
			patch_data = [p[2] for p in idx_and_patches]

			patch_mean_results = []
			patch_std_results  = []
			for patch in patch_data
				# Inference runs on GPU inside infer_model/infer_model_tta,
				# but aggregation is returned on host CPU for safe stitching.
				y_mean, y_std, _ = infer_model_tta(tstate_curr, model, patch, config)
				push!(patch_mean_results, y_mean)
				push!(patch_std_results,  y_std)
			end

			idx_and_y_mean = collect(zip(coordinates, patch_mean_results))
			idx_and_y_std  = collect(zip(coordinates, patch_std_results))

			# Reconstruct uses test_data_cpu dimensions
			y_mean = recreate_image_from_patches_test(
				idx_and_y_mean, padded_data_size, patch_size, size(test_data_cpu),
			)
			y_std = recreate_image_from_patches_test(
				idx_and_y_std, padded_data_size, patch_size, size(test_data_cpu),
			)

			# Apply optional post-processing to the MEAN prediction (std remains an uncertainty map)
			if get(config["learning"], "largest_connected_component", false)
				n_lcc = get(config["learning"], "n_lcc", 1)
				W, H, D, C_out, B = size(y_mean)
				y_pred_lcc = zeros(Float32, W, H, D, C_out, B)

				for b in 1:B
					for c in 1:C_out
						binary_mask = Array{Int32}(
							Array(y_mean[:, :, :, c, b]) .>= 0.5f0,
						)
						components = largest_connected_components(binary_mask, n_lcc)
						if !isempty(components)
							combined = reduce((a, b_) -> a .| b_, components)
							y_pred_lcc[:, :, :, c, b] = Float32.(combined)
						end
					end
				end
				y_mean = y_pred_lcc
			end

			# 3. EVALUATE ON HOST CPU: Compare against the CPU label
			metr = evaluate_metric(y_mean, test_label_cpu, config["learning"]["metric"])
			push!(test_metrics, metr)
			push!(results_mean, y_mean)
			push!(results_std,  y_std)
		end
	end
	return results_mean, results_std, test_metrics
end


"""
`pad_spatial_5d_with_medimage(image, target_size)`

Pads or crops a 5-D array to target_size along the spatial dims (1-3) using
MedImage's crop_or_pad. Channels (dim 4) and batch (dim 5) are iterated over
but never modified — only the spatial layout changes.
"""
function pad_spatial_5d_with_medimage(
	image::AbstractArray{T, 5},
	target_size::Tuple{Int, Int, Int},
) where T
	out = Array{T}(undef,
		target_size[1], target_size[2], target_size[3],
		size(image, 4), size(image, 5),
	)
	for n in axes(image, 5)
		for c in axes(image, 4)
			mi = medimage_from_array(view(image,:,:,:,c,n))
			padded = crop_or_pad(
				mi, target_size; interpolator = Nearest_neighbour_en, pad_val = 0,
			).voxel_data
			out[:, :, :, c, n] = padded
		end
	end
	return out
end


"""
`divide_into_patches_test(image, patch_size)`

Divides a 5-D image [W, H, D, C, B] into non-overlapping spatial patches of
patch_size. The spatial dims are zero-padded to the nearest multiple of patch_size
if necessary. Channels (dim 4) and batch (dim 5) are preserved whole in every patch.

Returns:
- patches      : Vector of [(x, y, z), patch_view] tuples.
- padded_size  : Full size of the (possibly padded) image.
"""
function divide_into_patches_test(
	image::AbstractArray{T, 5},
	patch_size::Tuple{Int, Int, Int},
) where T
	println("Dividing image into patches...")
	println("Size of the image: ", size(image))

	pad_size = (
		(size(image, 1) % patch_size[1]) != 0 ? patch_size[1] - size(image, 1) % patch_size[1] : 0,
		(size(image, 2) % patch_size[2]) != 0 ? patch_size[2] - size(image, 2) % patch_size[2] : 0,
		(size(image, 3) % patch_size[3]) != 0 ? patch_size[3] - size(image, 3) % patch_size[3] : 0,
	)

	padded_image = if any(pad_size .> 0)
		target_size = (
			size(image, 1) + pad_size[1],
			size(image, 2) + pad_size[2],
			size(image, 3) + pad_size[3],
		)
		pad_spatial_5d_with_medimage(image, target_size)
	else
		image
	end

	patches = []
	for x in 1:patch_size[1]:size(padded_image, 1)
		for y in 1:patch_size[2]:size(padded_image, 2)
			for z in 1:patch_size[3]:size(padded_image, 3)
				patch = view(
						padded_image,
						(x:(x+patch_size[1]-1)),
						(y:(y+patch_size[2]-1)),
						(z:(z+patch_size[3]-1)),
						:,  # all channels (modalities in / class logits out)
						:,  # all batch elements
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

Reassembles a full prediction volume from per-patch outputs, then crops back to
original_size to remove any padding added in divide_into_patches_test.

The reconstruction buffer is allocated with original_size[4] channels and
original_size[5] batch elements so that dim 4 (class logits) and dim 5 (batch)
are preserved exactly as they came out of the model.
"""
function recreate_image_from_patches_test(
	coords_with_patches,
	padded_size,
	patch_size::Tuple{Int, Int, Int},
	original_size,
)
	println("Recreating image from patches...")

	# C_out (dim 4) comes from the model output, which may differ from the input
	# modality count C_in. Infer it from the first patch rather than original_size[4].
	# original_size[5] (batch) is the same for inputs and outputs.
	first_patch = first(p[2] for p in coords_with_patches)
	c_out = size(first_patch, 4)
	b = original_size[5]

	reconstructed = zeros(Float32,
		padded_size[1], padded_size[2], padded_size[3],
		c_out,  # C_out class logits (may differ from C_in input modalities)
		b,      # batch
	)

	for (coords, patch) in coords_with_patches
		x, y, z = coords
		reconstructed[
			x:(x+patch_size[1]-1),
			y:(y+patch_size[2]-1),
			z:(z+patch_size[3]-1),
			:,
			:,
		] = patch
	end

	# Crop spatial dims back to the original unpadded size.
	# dim 4 (c_out) and dim 5 (batch) are never cropped — they are already correct.
	final_image = reconstructed[
		1:original_size[1],
		1:original_size[2],
		1:original_size[3],
		:,  # all C_out class logits
		:,  # all batch elements
	]

	println("Size of the final image: ", size(final_image))
	return final_image
end


"""
`process_results_test(results, test_metrics, labels, config)`

Aggregates per-patch / per-augmentation predictions into a single final prediction
and returns the corresponding evaluation metric.

Replaces the previous broken implementation which referenced an undefined `metr`
variable and called a non-existent `largest_connected_components` (plural) function.

mean(results) averages element-wise over the outer results vector (one entry per
augmentation/ensemble member), preserving the full 5-D shape [W, H, D, C_out, B] —
dim 4 (class logits) is averaged across ensemble members, not collapsed.
"""
function process_results_test(results, test_metrics, labels, config)
	println("Processing results...")

	if isempty(results)
		@warn "process_results_test called with empty results vector."
		return nothing, Dict{Int, Float64}()
	end

	# Average predictions element-wise across all ensemble/augmentation members.
	# Preserves full 5-D shape [W, H, D, C_out, B] — dim 4 (class logits) is
	# averaged across members, not collapsed.
	y_pred = mean(results)

	# Optional largest-connected-component post-processing on the averaged prediction.
	# Applied here (after ensemble averaging) rather than per-patch, so the spatial
	# context of the full reconstructed volume is available.
	if get(config["learning"], "largest_connected_component", false)
		n_lcc = get(config["learning"], "n_lcc", 1)
		W, H, D, C_out, B = size(y_pred)
		y_pred_lcc = zeros(Float32, W, H, D, C_out, B)

		for b in 1:B
			for c in 1:C_out
				binary_mask = Array{Int32}(
					Array(y_pred[:, :, :, c, b]) .>= 0.5f0,
				)
				components = largest_connected_components(binary_mask, n_lcc)
				if !isempty(components)
					combined = reduce((a, b_) -> a .| b_, components)
					y_pred_lcc[:, :, :, c, b] = Float32.(combined)
				end
			end
		end
		y_pred = y_pred_lcc
		println("Applied largest_connected_components (n_lcc=$n_lcc) to averaged prediction.")
	end

	# Recompute metric on the final (possibly LCC-filtered) averaged prediction
	# using the ground-truth labels passed in from evaluate_test_set_test.
	metr = evaluate_metric(y_pred, labels, config["learning"]["metric"])

	println("Processed $(length(results)) result(s). Final metric: $metr")
	return y_pred, metr
end
