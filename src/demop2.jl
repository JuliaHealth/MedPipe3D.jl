# =============================================================================
# MedPipe3D Training Pipeline - Task02_Heart
# Continuation after HDF5 creation
# Steps: Config → Train → Validate → Test
# =============================================================================

try
	using MedPipe3D
catch
	include(joinpath(@__DIR__, "MedPipe3D.jl"))
	using .MedPipe3D
end

using JSON
using HDF5
using CUDA
using Random

# =============================================================================
# PATHS
# =============================================================================

pathToHDF5  = joinpath(homedir(), "heart_dataset.hdf5")
config_dir  = joinpath(homedir(), "demo")
config_path = joinpath(config_dir, "config.json")
mkpath(config_dir)

# =============================================================================
# STEP 5: CREATE CONFIGURATION (non-interactive, all defaults set manually)
# =============================================================================
# Instead of using the interactive create_config_extended, we build the config
# directly as a Dict and save it to JSON.

config = Dict(
	"data" => Dict(
		"batch_size"          => 1,
		"batch_complete"      => false,
		"channel_size_imgs"   => 1,       # MRI = 1 channel
		"channel_size_masks"  => 1,
		"resample_to_target"  => false,
		"resample_to_spacing" => "avg",
		"target_spacing"      => nothing,
		"resample_size"       => "avg",
		"standardization"     => false,
		"normalization"       => true,
		"has_mask"            => true,
	),
	"augmentation" => Dict(
		"order" => [],           # no augmentations for this demo
		"p_rand" => 0.5,
		"augmentations" => Dict(),
		"processing_unit" => "GPU",
	),
	"learning" => Dict(
		"invertible_augmentations" => false,
		"Train_Val_Test_JSON" => false,
		"n_cross_val" => false,
		"n_folds" => 1,
		"test_train_validation" => (0.6, 0.2, 0.2),  # 60% train, 20% val, 20% test
		"class_JSON_path" => false,
		"additional_JSON_path" => false,
		"shuffle" => true,
		"patch_probabilistic_oversampling" => true,
		"patch_size" => [32, 32, 32],     # 32³ patches (lower GPU memory)
		"oversampling_probability" => 0.5,              # 50% chance of foreground patch
		"metric" => "dice",
		"largest_connected_component" => false,
		"n_lcc" => nothing,
	),
	"model" => Dict(
		"optimizer_name" => "Adam",
		"optimizer_args" => "lr=0.001",
		"num_epochs" => 10,             # small for demo
		"early_stopping" => true,
		"patience" => 3,
		"early_stopping_min_delta" => 0.001,
		"early_stopping_metric" => "val_loss",
		"loss_function_name" => "dice"          # dice loss for segmentation
	),
)

# Save config to JSON
open(config_path, "w") do f
	print(f, JSON.json(config, 4))
end
println("Config saved to: ", config_path)

# =============================================================================
# STEP 6: VERIFY HDF5 AND CONFIG BEFORE TRAINING
# =============================================================================

println("\n--- Verifying HDF5 file ---")
h5open(pathToHDF5, "r") do h5
	patient_keys = keys(h5)
	println("Number of patients in HDF5: ", length(patient_keys))
	# Check first patient structure
	first = h5[patient_keys[1]]
	println("First patient groups: ", keys(first))
	println("Image shape: ", size(read(first["images/data"])))
	println("Mask shape : ", size(read(first["masks/data"])))
	println("Mask unique values: ", unique(read(first["masks/data"])))
end

# =============================================================================
# STEP 7: RUN TRAINING LOOP
# =============================================================================

println("\n--- Starting Training ---")
println("HDF5 path  : ", pathToHDF5)
println("Config path: ", config_path)
println("Epochs     : ", config["model"]["num_epochs"])
println("Patch size : ", config["learning"]["patch_size"])
println("GPU available: ", CUDA.functional())

rng_seed = 42

# main_loop handles:
#   - train/val/test split
#   - batch loading from HDF5
#   - patch extraction with oversampling
#   - model creation (U-Net style segmentation)
#   - training with Adam optimizer
#   - validation with Dice metric
#   - early stopping
final_state = main_loop(pathToHDF5, config_path, rng_seed)

println("\n--- Training Complete ---")
println("Final training state: ", typeof(final_state))
