using Test
using HDF5
using Random



# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
Create a minimal in-memory HDF5 file with synthetic image/mask data.

Layout per group:
  <group>/images/data   → Float32 array of shape (W, H, D, C)
  <group>/masks/data    → Float32 array of shape (W, H, D, C)
  <group>/images/metadata/<meta_key>  → HDF5 group with attribute "class"

Arguments
- `path`         : file path for the temp HDF5 file
- `groups`       : vector of group name strings (e.g. ["patient_01"])
- `image_shape`  : (W, H, D, C) tuple
- `class_names`  : parallel vector of class strings per group (e.g. ["2_liver"])
				   pass `nothing` to skip writing metadata/class attributes
- `label_val`    : scalar value written into every mask voxel (default 1)
"""
function make_test_h5(path, groups, image_shape; class_names = nothing, label_val = 1)
	h5open(path, "w") do h5
		for (i, group) in enumerate(groups)
			img  = rand(Float32, image_shape)
			mask = fill(Float32(label_val), image_shape)

			h5[group*"/images/data"] = img
			h5[group*"/masks/data"]  = mask

			if class_names !== nothing
				meta_path = group * "/images/metadata/meta_0"
				create_group(h5, meta_path)
				attributes(h5[meta_path])["class"] = class_names[i]
			end
		end
	end
	return path
end

"""Build a minimal config dict used by most tests."""
function base_config(; has_mask = true, class_json = false, oversampling = false,
	patch_size = (8, 8, 8), oversampling_prob = 0.5,
	processing_unit = "CPU")
	Dict(
		"data" => Dict("has_mask" => has_mask),
		"learning" => Dict(
			"class_JSON_path"                  => class_json,
			"patch_probabilistic_oversampling" => oversampling,
			"patch_size"                       => collect(patch_size),
			"oversampling_probability"         => oversampling_prob,
		),
		"augmentation" => Dict("processing_unit" => processing_unit),
	)
end

# Shared temp file path — each test writes its own data
const TMP_H5 = tempname() * ".h5"

# Clean up after the full suite
function cleanup()
	isfile(TMP_H5) && rm(TMP_H5)
end


# ──────────────────────────────────────────────────────────────────────────────
# get_random_patch
# ──────────────────────────────────────────────────────────────────────────────
@testset "get_random_patch" begin

	@testset "output shape matches patch_size when image is larger" begin
		image = rand(Float32, 20, 20, 20)
		label = rand(Float32, 20, 20, 20)
		patch_size = (8, 8, 8)
		img_p, lbl_p = get_random_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end

	@testset "image and label patch are spatially consistent" begin
		# Use a label with a unique value at a known location so we can verify
		# the same spatial region was cut from both arrays.
		image = zeros(Float32, 16, 16, 16)
		label = zeros(Float32, 16, 16, 16)
		image[5, 5, 5] = 99.0f0
		label[5, 5, 5] = 99.0f0
		patch_size = (4, 4, 4)

		# Run many times; at least one run must capture the marked voxel in both
		found_consistent = false
		for _ in 1:200
			ip, lp = get_random_patch(image, label, patch_size)
			if any(ip .== 99.0f0)
				@test any(lp .== 99.0f0)
				found_consistent = true
				break
			end
		end
		# The marked voxel must appear in both or neither — never split between them
		@test found_consistent
	end

	@testset "patch fits inside image boundaries (no out-of-bounds)" begin
		image = rand(Float32, 10, 10, 10)
		label = rand(Float32, 10, 10, 10)
		patch_size = (5, 5, 5)
		for _ in 1:50
			img_p, lbl_p = get_random_patch(image, label, patch_size)
			@test size(img_p) == patch_size
			@test size(lbl_p) == patch_size
		end
	end

	@testset "image smaller than patch_size is padded to fit" begin
		image = rand(Float32, 4, 4, 4)
		label = rand(Float32, 4, 4, 4)
		patch_size = (8, 8, 8)
		img_p, lbl_p = get_random_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end

	@testset "exact-fit image (patch_size == image size)" begin
		image = rand(Float32, 8, 8, 8)
		label = rand(Float32, 8, 8, 8)
		patch_size = (8, 8, 8)
		img_p, lbl_p = get_random_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end

	@testset "non-cubic patch_size" begin
		image = rand(Float32, 20, 16, 12)
		label = rand(Float32, 20, 16, 12)
		patch_size = (10, 8, 6)
		img_p, lbl_p = get_random_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# extract_nonzero_patch
# ──────────────────────────────────────────────────────────────────────────────
@testset "extract_nonzero_patch" begin

	@testset "patch contains at least one non-zero label voxel (foreground present)" begin
		image = rand(Float32, 20, 20, 20)
		label = zeros(Float32, 20, 20, 20)
		label[10, 10, 10] = 1.0f0
		patch_size = (6, 6, 6)

		# The center of the patch is on a non-zero voxel, so the patch must contain it
		img_p, lbl_p = extract_nonzero_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
		@test any(lbl_p .!= 0)
	end

	@testset "falls back to random patch when label is all zeros" begin
		image = rand(Float32, 20, 20, 20)
		label = zeros(Float32, 20, 20, 20)
		patch_size = (8, 8, 8)
		img_p, lbl_p = extract_nonzero_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end

	@testset "output shapes match patch_size" begin
		image = rand(Float32, 16, 16, 16)
		label = zeros(Float32, 16, 16, 16)
		label[8, 8, 8] = 2.0f0
		patch_size = (4, 4, 4)
		img_p, lbl_p = extract_nonzero_patch(image, label, patch_size)
		@test size(img_p) == patch_size
		@test size(lbl_p) == patch_size
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# extract_patch  (probability routing)
# ──────────────────────────────────────────────────────────────────────────────
@testset "extract_patch routing" begin

	@testset "probability=1.0 always calls extract_nonzero_patch path" begin
		config = base_config(oversampling_prob = 1.0)
		image = rand(Float32, 20, 20, 20)
		label = zeros(Float32, 20, 20, 20)
		label[10, 10, 10] = 1.0f0
		patch_size = (6, 6, 6)

		for _ in 1:10
			img_p, lbl_p = extract_patch(image, label, patch_size, config)
			# With prob=1.0 the nonzero path is always taken; patch must contain foreground
			@test any(lbl_p .!= 0)
		end
	end

	@testset "probability=0.0 always calls get_random_patch path" begin
		config = base_config(oversampling_prob = 0.0)
		image = rand(Float32, 20, 20, 20)
		label = zeros(Float32, 20, 20, 20)   # all-zero label → random patch path always
		patch_size = (6, 6, 6)

		for _ in 1:10
			img_p, lbl_p = extract_patch(image, label, patch_size, config)
			@test size(img_p) == patch_size
			@test size(lbl_p) == patch_size
		end
	end

	@testset "output shape is always patch_size regardless of routing" begin
		config = base_config(oversampling_prob = 0.5)
		image = rand(Float32, 20, 20, 20)
		label = zeros(Float32, 20, 20, 20)
		label[10, 10, 10] = 1.0f0
		patch_size = (8, 8, 8)

		for _ in 1:20
			img_p, lbl_p = extract_patch(image, label, patch_size, config)
			@test size(img_p) == patch_size
			@test size(lbl_p) == patch_size
		end
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# get_batch_with_classes
# ──────────────────────────────────────────────────────────────────────────────
@testset "get_batch_with_classes" begin

	@testset "binary mode — single group, no class JSON" begin
		shape = (10, 10, 10, 2)   # (W, H, D, C)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, class_json = false)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, class_labels = get_batch_with_classes(["patient_01"], h5, config)
			@test ndims(imgs) == 5
			@test ndims(lbls) == 5
			@test size(imgs, 5) == 1   # one group → batch dim = 1
			@test class_labels == [1]
		end
		rm(TMP_H5)
	end

	@testset "binary mode — multiple groups concatenated on dim 5" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1", "p2", "p3"], shape)
		config = base_config(has_mask = true, class_json = false)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_batch_with_classes(["p1", "p2", "p3"], h5, config)
			@test size(imgs, 5) == 3
			@test size(lbls, 5) == 3
		end
		rm(TMP_H5)
	end

	@testset "image tensor spatial dims match raw data" begin
		shape = (12, 14, 16, 2)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, class_json = false)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_batch_with_classes(["patient_01"], h5, config)
			@test size(imgs)[1:4] == shape
		end
		rm(TMP_H5)
	end

	@testset "has_mask=false — labels tensor is empty" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = false, class_json = false)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_batch_with_classes(["patient_01"], h5, config)
			@test ndims(imgs) == 5
			# labels array should be empty when has_mask is false
			@test isempty(lbls)
		end
		rm(TMP_H5)
	end

	@testset "multi-class — label values are scaled by class index" begin
		shape = (8, 8, 8, 1)
		# class string format expected by get_class_labels: "<idx>_<name>"
		make_test_h5(TMP_H5, ["patient_01"], shape;
			class_names = ["2_liver"], label_val = 1)
		config = base_config(has_mask = true, class_json = "/some/path/classes.json")

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_batch_with_classes(["patient_01"], h5, config)
			# Original label_val=1, class_idx=2 → all foreground voxels should be 2
			@test all(lbls[lbls .!= 0] .== 2.0f0)
		end
		rm(TMP_H5)
	end

	@testset "multi-class — class_labels dict is populated" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["patient_01", "patient_02"], shape;
			class_names = ["1_kidney", "2_liver"])
		config = base_config(has_mask = true, class_json = "/some/path/classes.json")

		h5open(TMP_H5, "r") do h5
			_, _, class_labels = get_batch_with_classes(["patient_01", "patient_02"], h5, config)
			@test !isempty(class_labels)
		end
		rm(TMP_H5)
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# get_patch_batch_with_classes
# ──────────────────────────────────────────────────────────────────────────────
@testset "get_patch_batch_with_classes" begin

	@testset "output patch spatial dims match configured patch_size" begin
		shape = (20, 20, 20, 2)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		patch_size = (8, 8, 8)
		config = base_config(has_mask = true, oversampling = true,
			patch_size = patch_size, oversampling_prob = 0.5)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_patch_batch_with_classes(["patient_01"], h5, config)
			@test size(imgs)[1:3] == patch_size
		end
		rm(TMP_H5)
	end

	@testset "batch dim (dim 5) equals number of input groups" begin
		shape = (16, 16, 16, 1)
		make_test_h5(TMP_H5, ["p1", "p2"], shape)
		config = base_config(has_mask = true, oversampling = true,
			patch_size = (8, 8, 8), oversampling_prob = 0.0)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_patch_batch_with_classes(["p1", "p2"], h5, config)
			@test size(imgs, 5) == 2
			@test size(lbls, 5) == 2
		end
		rm(TMP_H5)
	end

	@testset "channel dim (dim 4) preserved from raw data" begin
		n_channels = 3
		shape = (16, 16, 16, n_channels)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, oversampling = true,
			patch_size = (8, 8, 8), oversampling_prob = 0.0)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, _ = get_patch_batch_with_classes(["patient_01"], h5, config)
			@test size(imgs, 4) == n_channels
		end
		rm(TMP_H5)
	end

	@testset "no class JSON → class_labels defaults to [1]" begin
		shape = (16, 16, 16, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, class_json = false,
			patch_size = (8, 8, 8), oversampling_prob = 0.0)

		h5open(TMP_H5, "r") do h5
			_, _, class_labels = get_patch_batch_with_classes(["patient_01"], h5, config)
			@test class_labels == [1]
		end
		rm(TMP_H5)
	end

	@testset "multi-class — label values scaled by class index" begin
		shape = (16, 16, 16, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape;
			class_names = ["3_spleen"], label_val = 1)
		config = base_config(has_mask = true, class_json = "/path/classes.json",
			patch_size = (8, 8, 8), oversampling_prob = 0.0)

		h5open(TMP_H5, "r") do h5
			_, lbls, class_labels = get_patch_batch_with_classes(["patient_01"], h5, config)
			@test only(class_labels) == 3
			@test all(lbls[lbls .!= 0] .== 3.0f0)
		end
		rm(TMP_H5)
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# get_class_labels
# ──────────────────────────────────────────────────────────────────────────────
@testset "get_class_labels" begin

	@testset "returns correct class string from image metadata" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape; class_names = ["2_liver"])
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			result = get_class_labels(["patient_01"], h5, config)
			@test haskey(result, "2_liver")
			@test result["2_liver"] == ["patient_01"]
		end
		rm(TMP_H5)
	end

	@testset "groups multiple patients under the same class key" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1", "p2"], shape; class_names = ["1_kidney", "1_kidney"])
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			result = get_class_labels(["p1", "p2"], h5, config)
			@test haskey(result, "1_kidney")
			@test length(result["1_kidney"]) == 2
		end
		rm(TMP_H5)
	end

	@testset "no class JSON → defaults to 'class1' key" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape; class_names = ["2_liver"])
		config = base_config(class_json = false)

		h5open(TMP_H5, "r") do h5
			result = get_class_labels(["patient_01"], h5, config)
			@test haskey(result, "class1")
		end
		rm(TMP_H5)
	end

	@testset "missing metadata group → patient is not added to result" begin
		shape = (8, 8, 8, 1)
		# Write a group with no metadata subgroup at all
		h5open(TMP_H5, "w") do h5
			h5["patient_01/images/data"] = rand(Float32, shape...)
			h5["patient_01/masks/data"]  = ones(Float32, shape...)
		end
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			result = get_class_labels(["patient_01"], h5, config)
			@test isempty(result)
		end
		rm(TMP_H5)
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# get_class_mapping
# ──────────────────────────────────────────────────────────────────────────────
@testset "get_class_mapping" begin

	@testset "each group is assigned a unique indexed key" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1", "p2"], shape;
			class_names = ["2_liver", "3_spleen"])
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			mapping = get_class_mapping(["p1", "p2"], h5, config)
			@test length(mapping) == 2
			all_groups = vcat(values(mapping)...)
			@test "p1" in all_groups
			@test "p2" in all_groups
		end
		rm(TMP_H5)
	end

	@testset "same class name gets same key — patients grouped together" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1", "p2", "p3"], shape;
			class_names = ["1_kidney", "1_kidney", "2_liver"])
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			mapping = get_class_mapping(["p1", "p2", "p3"], h5, config)
			kidney_key = only(k for k in keys(mapping) if endswith(k, "_kidney"))
			@test length(mapping[kidney_key]) == 2
		end
		rm(TMP_H5)
	end

	@testset "keys are formatted as '<index>_<classname>'" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1"], shape; class_names = ["2_liver"])
		config = base_config(class_json = "/path/classes.json")

		h5open(TMP_H5, "r") do h5
			mapping = get_class_mapping(["p1"], h5, config)
			key = only(keys(mapping))
			parts = split(key, "_", limit = 2)
			@test length(parts) == 2
			@test !isnothing(tryparse(Int, parts[1]))   # first part is an integer index
		end
		rm(TMP_H5)
	end
end


# ──────────────────────────────────────────────────────────────────────────────
# fetch_and_preprocess_data  (integration)
# ──────────────────────────────────────────────────────────────────────────────
@testset "fetch_and_preprocess_data integration" begin

	@testset "CPU path — full volume, binary" begin
		shape = (10, 10, 10, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, oversampling = false)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, unique_classes = fetch_and_preprocess_data(["patient_01"], h5, config)
			@test ndims(imgs) == 5
			@test ndims(lbls) == 5
			@test unique_classes == [1]
		end
		rm(TMP_H5)
	end

	@testset "CPU path — patch mode, oversampling enabled" begin
		shape = (20, 20, 20, 1)
		make_test_h5(TMP_H5, ["patient_01"], shape)
		config = base_config(has_mask = true, oversampling = true,
			patch_size = (8, 8, 8), oversampling_prob = 0.5)

		h5open(TMP_H5, "r") do h5
			imgs, lbls, unique_classes = fetch_and_preprocess_data(["patient_01"], h5, config)
			@test size(imgs)[1:3] == (8, 8, 8)
		end
		rm(TMP_H5)
	end

	@testset "unique_classes contains only distinct values" begin
		shape = (8, 8, 8, 1)
		make_test_h5(TMP_H5, ["p1", "p2"], shape)
		config = base_config(has_mask = true, oversampling = false)

		h5open(TMP_H5, "r") do h5
			_, _, unique_classes = fetch_and_preprocess_data(["p1", "p2"], h5, config)
			@test length(unique_classes) == length(unique(unique_classes))
		end
		rm(TMP_H5)
	end
end


# Final cleanup
cleanup()