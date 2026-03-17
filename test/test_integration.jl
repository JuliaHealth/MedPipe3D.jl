ENV["MEDPIPE_INTEGRATION_TESTS"] = "true"
ENV["MEDPIPE_USE_GPU"] = "true"

using Test
using HDF5
using JSON
using Random
using Dates
using MedImages

# ── Paths to real data ────────────────────────────────────────
const DATASET_ROOT = expanduser("../dataset")
const HDF5_PATH    = joinpath(DATASET_ROOT, "HDF5", "heart_dataset.hdf5")
const CONFIG_PATH  = joinpath(DATASET_ROOT, "config", "config.json")
const NIFTI_DIR    = joinpath(DATASET_ROOT, "nifti_output")
const RAW_DIR      = joinpath(DATASET_ROOT, "raw", "Task02_Heart")

# Known patients present in nifti_output
const NIFTI_PATIENTS = ["la_022", "la_023", "la_024", "la_026", "la_029", "la_030"]

# ── Guards ────────────────────────────────────────────────────
run_integration() = get(ENV, "MEDPIPE_INTEGRATION_TESTS", "false") == "true"
use_gpu()         = get(ENV, "MEDPIPE_USE_GPU", "false") == "true"

data_available() = isfile(HDF5_PATH) && isfile(CONFIG_PATH)

lux_ok = try
	using Lux, MLDataDevices, Optimisers
	true
catch
	false
end
"""
`main_loop_test_1(hdf5_path, config_path, rng_seed)`

Mock version of `main_loop` designed specifically for integration testing. 
It initializes the model, runs a shortened training loop, and returns the 
components needed for downstream test-set evaluation.
"""
function main_loop_test_1(hdf5_path, config_path, rng_seed)
	println("Running main_loop_test_1 setup...")
	config = JSON.parsefile(config_path)
	rng = Xoshiro(rng_seed)

	# Pre-allocate return variables
	test_groups = String[]
	model = nothing
	final_tstate = nothing

	h5open(hdf5_path, "r") do h5
		# 1. Get data splits
		indices_dict = proc_hdf5(h5, config, rng)

		# Safely extract a test group (fallback to validation if 'test' isn't explicitly split)
		test_groups = haskey(indices_dict, "test") ? indices_dict["test"] : indices_dict["validation"]

		# 2. Extract dimensions and classes
		image_data = read(h5[indices_dict["train"][1]*"/images/data"])
		n_channels = size(image_data, 4)
		classes = get_class_labels(indices_dict["train"], h5, config)
		num_classes = length(classes) + 1

		# 3. Initialize Model and State
		model = create_segmentation_model(num_classes, n_channels)
		optimizer = get_optimiser(config["model"]["optimizer_name"])
		loss_fn = get_loss_function(config["model"]["loss_function_name"])

		# Check environment for GPU usage
		use_gpu = get(ENV, "MEDPIPE_USE_GPU", "false") == "true"
		tstate = initialize_train_state(rng, model, optimizer; use_gpu = use_gpu)

		# 4. Run a fast epoch loop (capped at 2 epochs for testing speed)
		final_tstate = epoch_loop(
			2, indices_dict["train"], indices_dict["validation"],
			h5, model, tstate, config, loss_fn, num_classes,
		)
	end

	# Return exactly what test_integration.jl expects:
	# test_groups, model, final_tstate, cfg
	return test_groups, model, final_tstate, config
end


"""
`main_loop_test_2(test_groups, hdf5_path, model, tstate, config)`

Mock version of the test-set evaluation phase. Takes the outputs from 
`main_loop_test_1` and evaluates the test set, returning the final state and metrics.
"""
function main_loop_test_2(test_groups, hdf5_path, model, tstate, config)
	println("Running main_loop_test_2 evaluation...")
	test_metrics = []

	h5open(hdf5_path, "r") do h5
		# Call your newly fixed evaluate_test_set_test function
		test_metrics = evaluate_test_set_test(test_groups, h5, model, tstate, config)
	end

	# Return exactly what test_integration.jl expects:
	# final_tstate, test_metrics
	return tstate, test_metrics
end
# ── Full MedImage constructor helper ──────────────────────────
# Covers all 19 required fields so tests don't need a real file
# just to build a MedImage for saving/round-trip checks.
function _make_medimage(arr::AbstractArray{Float32};
	origin::NTuple{3, Float64}    = (0.0, 0.0, 0.0),
	spacing::NTuple{3, Float64}   = (1.0, 1.0, 1.0),
	direction::NTuple{9, Float64} = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)::MedImage
	MedImage(
		arr,                                                          # voxel_data
		origin,                                                       # origin
		spacing,                                                      # spacing
		direction,                                                    # direction
		first(instances(MedImages.Image_type)),                       # image_type
		first(instances(MedImages.Image_subtype)),                    # image_subtype
		now(),                                                        # date_of_saving
		now(),                                                        # acquistion_time
		"test_patient",                                               # patient_id
		first(instances(MedImages.current_device_enum)),              # current_device
		"1.2.3.test",                                                 # study_uid
		"patient_test_001",                                           # patient_uid
		"1.2.3.series",                                               # series_uid
		"test study",                                                 # study_description
		"test.nii.gz",                                                # legacy_file_name
		Dict{Any, Any}(),                                             # display_data
		Dict{Any, Any}(),                                             # clinical_data
		false,                                                        # is_contrast_administered
		Dict{Any, Any}(),                                              # metadata
	)
end

# ── Tests ─────────────────────────────────────────────────────

@testset "integration — real heart dataset" begin

	if !run_integration()
		@info "Skipping: set MEDPIPE_INTEGRATION_TESTS=true to run integration tests"
	elseif !data_available()
		@info "Skipping: real data not found at $DATASET_ROOT"
	else

		config = JSON.parsefile(CONFIG_PATH)
		rng    = Xoshiro(42)

		# ── Config sanity ─────────────────────────────────────
		@testset "config structure" begin
			for section in ("data", "augmentation", "learning", "model")
				@test haskey(config, section)
			end
			@test config["data"]["channel_size_imgs"] isa Int
			@test config["data"]["channel_size_masks"] isa Int
		end

		# ── HDF5 structure ────────────────────────────────────
		@testset "HDF5 file structure" begin
			h5open(HDF5_PATH, "r") do h5
				@test length(keys(h5)) > 0

				# Every top-level group should have images and masks
				for batch_key in keys(h5)
					bg = h5[batch_key]
					isa(bg, HDF5.Group) || continue
					@test haskey(bg, "images")
					@test haskey(bg, "masks")

					# images/data should be 4-D or 5-D
					img_obj = bg["images"]
					if isa(img_obj, HDF5.Group) && haskey(img_obj, "data")
						nd = ndims(img_obj["data"])
						@test nd ∈ (4, 5)
					end
				end
			end
		end

		# ── proc_hdf5 splits ──────────────────────────────────
		@testset "proc_hdf5 produces valid splits" begin
			h5open(HDF5_PATH, "r") do h5
				splits = proc_hdf5(h5, config, rng)

				@test haskey(splits, "train")
				@test haskey(splits, "validation")
				@test !isempty(splits["train"])
				@test !isempty(splits["validation"])

				# All paths must exist in the file
				for key in vcat(values(splits)...)
					@test haskey(h5, key)
				end

				# No overlap between splits
				all_keys = vcat(values(splits)...)
				@test length(all_keys) == length(unique(all_keys))
			end
		end

		# ── get_batch_with_classes ────────────────────────────
		@testset "get_batch_with_classes shape and types" begin
			h5open(HDF5_PATH, "r") do h5
				splits = proc_hdf5(h5, config, rng)
				images, labels, classes = get_batch_with_classes(
					splits["train"], h5, config)

				@test ndims(images) == 5
				@test ndims(labels) == 5
				@test eltype(images) <: AbstractFloat
				@test size(images)[1:3] == size(labels)[1:3]   # spatial dims match
				@test size(images, 5) == size(labels, 5)     # batch dim matches
				@test length(classes) == size(images, 5)
				@test !isempty(classes)
			end
		end

		# ── get_class_labels ──────────────────────────────────
		@testset "get_class_labels uniqueness" begin
			h5open(HDF5_PATH, "r") do h5
				splits = proc_hdf5(h5, config, rng)
				labels = get_class_labels(splits["train"], h5, config)

				@test !isempty(labels)
				@test length(labels) == length(unique(labels))
			end
		end

		# ── NIfTI output files exist ──────────────────────────
		@testset "nifti_output files present" begin
			for patient in NIFTI_PATIENTS
				img_file  = joinpath(NIFTI_DIR, "$(patient).nii_image_after.gz.nii.gz")
				mask_file = joinpath(NIFTI_DIR, "$(patient).nii_mask_after.gz.nii.gz")

				if !isfile(img_file)
					println("Missing image: $img_file")
				end
				@test isfile(img_file)

				if !isfile(mask_file)
					println("Missing mask: $mask_file")
				end
				@test isfile(mask_file)
			end
		end

		# ── process_and_save_medimage_test round-trip ─────────
		@testset "process_and_save_medimage_test round-trip" begin
			# Use the first real NIfTI image as source
			src_path = joinpath(NIFTI_DIR, "la_022.nii_image_after.gz.nii.gz")

			mktempdir() do out_folder
				original  = MedImages.load_image(src_path, "test_patient")
				pred_data = rand(Float32, size(original.voxel_data))
				meta      = Dict{String, Any}("file_path" => src_path)

				temp_h5_path = joinpath(out_folder, "test_roundtrip.hdf5")
				group_name = "test_predictions"

				h5open(temp_h5_path, "w") do h5
					# Call the updated function with HDF5 file handle
					dataset_uuid = process_and_save_medimage_test(meta, pred_data, h5, group_name)

					# Verify the parent group was created in the HDF5 file
					@test haskey(h5, group_name)

					# Verify the specific UUID dataset/group was saved inside it
					pred_group = h5[group_name]
					@test haskey(pred_group, dataset_uuid)

					# Verify the voxel data size inside the HDF5 file matches our prediction
					saved_uuid_group = pred_group[dataset_uuid]
					if haskey(saved_uuid_group, "voxel_data")
						saved_data = read(saved_uuid_group["voxel_data"])
						@test size(saved_data) == size(pred_data)
					end

					# Note: If MedImages.jl exports a specific HDF5 load function 
					# (like `load_med_image(h5, path)`), you could reload it here 
					# to strictly verify `spacing`, `origin`, and `direction`.
				end
			end
		end

		# ── save_results_test ─────────────────────────────────
		# Verifies that save_results_test writes a NIfTI file to the
		# output folder and that the saved file is loadable.
		@testset "save_results_test writes output to HDF5" begin
			mktempdir() do out_folder
				# Build a synthetic MedImage so we don't need a real file on disk
				arr = rand(Float32, 8, 8, 8)
				img = _make_medimage(arr)

				# Write it as a NIfTI so process_and_save_medimage_test can load it to copy metadata
				src_path = joinpath(out_folder, "source.nii.gz")
				create_nii_from_medimage(img, src_path)

				# Simulate a model prediction (same shape as voxel data)
				y_pred = rand(Float32, size(arr))
				test_config = Dict{String, Any}()

				# Create a temporary HDF5 file to act as our pipeline output
				temp_h5_path = joinpath(out_folder, "test_output.hdf5")

				h5open(temp_h5_path, "w") do h5
					# Execute the new HDF5 save function
					save_results_test(y_pred, Dict{String, Any}("file_path" => src_path), test_config, h5)

					# 1. Verify the parent group was created
					@test haskey(h5, "test_predictions")

					# 2. Verify a UUID dataset/group was added inside it
					pred_group = h5["test_predictions"]
					uuid_keys = keys(pred_group)
					@test length(uuid_keys) == 1

					# 3. Verify the actual data matches our prediction dimensions
					# (Assuming save_med_image stores the array under a "voxel_data" key)
					saved_uuid_group = pred_group[uuid_keys[1]]
					if haskey(saved_uuid_group, "voxel_data")
						saved_data = read(saved_uuid_group["voxel_data"])
						@test size(saved_data) == size(arr)
					end
				end
			end
		end
		@testset "evaluate_validation smoke" begin
			if !lux_ok
				@info "Skipping: Lux not available"
			else
				h5open(HDF5_PATH, "r") do h5
					splits   = proc_hdf5(h5, config, rng)
					img_data = read(h5[splits["train"][1]*"/images/data"])

					n_channels  = size(img_data, 4)
					classes     = get_class_labels(splits["train"], h5, config)
					num_classes = length(classes) + 1

					model     = create_segmentation_model(num_classes, n_channels)
					optimizer = get_optimiser(config["model"]["optimizer_name"])
					loss_fn   = get_loss_function(config["model"]["loss_function_name"])
					tstate    = initialize_train_state(rng, model, optimizer; use_gpu = use_gpu())

					@test_nowarn evaluate_validation(
						splits["validation"], h5, model, tstate,
						loss_fn, config, num_classes)
				end
			end
		end

		# ── epoch_loop (2 epochs, early stopping enabled) ─────
		@testset "epoch_loop runs and returns tstate" begin
			if !lux_ok
				@info "Skipping: Lux not available"
			else
				h5open(HDF5_PATH, "r") do h5
					splits   = proc_hdf5(h5, config, rng)
					img_data = read(h5[splits["train"][1]*"/images/data"])

					n_channels  = size(img_data, 4)
					classes     = get_class_labels(splits["train"], h5, config)
					num_classes = length(classes) + 1

					model     = create_segmentation_model(num_classes, n_channels)
					optimizer = get_optimiser(config["model"]["optimizer_name"])
					loss_fn   = get_loss_function(config["model"]["loss_function_name"])
					tstate    = initialize_train_state(rng, model, optimizer; use_gpu = use_gpu())

					# Run only 2 epochs regardless of config to keep test fast
					final_tstate = epoch_loop(
						2,
						splits["train"], splits["validation"],
						h5, model, tstate, config, loss_fn, num_classes)

					@test final_tstate !== nothing
				end
			end
		end

		# ── Full main_loop end-to-end ──────────────────
		@testset "main_loop end-to-end" begin
			if !lux_ok
				@info "Skipping: Lux not available"
			else
				test_groups, model, final_tstate, cfg =
					main_loop_test_1(HDF5_PATH, CONFIG_PATH, 42)

				@test !isempty(test_groups)
				@test model !== nothing
				@test final_tstate !== nothing
				@test cfg !== nothing
			end
		end

		# ── main_loop test-set evaluation ──────────────
		@testset "main_loop test-set evaluation" begin
			if !lux_ok
				@info "Skipping: Lux not available"
			else
				test_groups, model, tstate, cfg =
					main_loop_test_1(HDF5_PATH, CONFIG_PATH, 42)

				final_tstate, test_metrics =
					main_loop_test_2(test_groups, HDF5_PATH, model, tstate, cfg)

				@test final_tstate !== nothing
				@test test_metrics !== nothing
			end
		end

	end  # data_available

end  # @testset
