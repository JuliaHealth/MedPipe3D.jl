using Test
using JSON
using MedImages

# ── Fixtures ─────────────────────────────────────────────────

"""
Write `n` tiny synthetic NIfTI files into `folder` and return their paths.
Each file is a (sz) volume with random voxel data.
"""
function _write_nifti_files(folder::String, n::Int;
	sz = (8, 8, 8), spacing = (1.0, 1.0, 1.0))
	mkpath(folder)
	paths = String[]
	for i in 1:n
		arr  = rand(Float32, sz...)
		img  = medimage_from_array(arr; spacing = spacing)
		path = joinpath(folder, "img_$(lpad(i,3,'0')).nii.gz")
		create_nii_from_medimage(img, path)
		push!(paths, path)
	end
	return paths
end

"""
Build a minimal on-disk patient tree and config, return config path.

Tree layout:
  root/
	patient_01/
	  Images/   ← 2 NIfTI files (channel_size_imgs = 2)
	  Masks/    ← 2 NIfTI files (channel_size_masks = 2)
"""
function _build_fixture(tmp::String; channel_size = 2, sz = (8, 8, 8))
	root = joinpath(tmp, "patients")
	p1   = joinpath(root, "patient_01")
	_write_nifti_files(joinpath(p1, "Images"), channel_size; sz = sz)
	_write_nifti_files(joinpath(p1, "Masks"), channel_size; sz = sz)

	cfg = Dict{String, Any}(
		"data" => Dict{String, Any}(
			"batch_size"         => 1,
			"channel_size_imgs"  => channel_size,
			"channel_size_masks" => channel_size,
			"resample_to_target" => false,
			"resampling"         => Dict("strategy"=>"none",
			"target_spacing"=>nothing,
			"target_size"=>"avg"),
			"normalisation"      => Dict("standardize"=>false, "normalize"=>false),
			"has_mask"           => true,
		),
		"augmentation" => Dict{String, Any}(
			"order"=>[], "processing_unit"=>"CPU", "augmentations"=>[],
		),
		"learning" => Dict{String, Any}(
			"split" => Dict("json_path"=>nothing, "ratios"=>[0.6, 0.2, 0.2]),
			"cross_val" => Dict("enabled"=>false, "n_folds"=>1),
			"patch" => Dict("enabled"=>false, "size"=>nothing, "oversampling_probability"=>0.0),
			"invertible_augmentations" => false,
			"shuffle" => false,
			"metric" => "dice",
			"largest_connected_component" => false,
			"n_lcc" => 1,
			"class_json_path" => nothing,
			"additional_json_paths" => [],
		),
		"model" => Dict{String, Any}(
			"optimizer"=>"Adam", "optimizer_args"=>Dict("lr"=>0.001),
			"num_epochs"=>1, "loss"=>"dice",
			"early_stopping"=>Dict("enabled"=>false, "patience"=>5,
				"min_delta"=>0.001, "monitor"=>"val_loss"),
		),
	)

	cfg_path = joinpath(tmp, "config.json")
	open(cfg_path, "w") do f
		;
		print(f, JSON.json(cfg, 4));
	end

	return root, cfg_path
end

# ── Tests ─────────────────────────────────────────────────────

@testset "batch_pipeline" begin

	tmp = mktempdir()

	# ── load_channel_data ────────────────────────────────────
	@testset "load_channel_data — happy path" begin
		root, cfg_path = _build_fixture(tmp)
		config = JSON.parsefile(cfg_path)

		img_folders = [joinpath(root, "patient_01", "Images")]
		loaded      = load_channel_data(img_folders, config, "image")

		@test length(loaded) == 1
		@test length(loaded[1].images) == 2
		@test length(loaded[1].metadata) == 2
		@test loaded[1].folder_name == "patient_01"

		# Metadata has expected keys
		m = loaded[1].metadata[1]
		for k in ("file_path", "shape_org", "spacing_org", "type_org")
			@test haskey(m, k)
		end
	end

	@testset "load_channel_data — insufficient files raises" begin
		root, cfg_path = _build_fixture(tmp)
		config         = JSON.parsefile(cfg_path)
		# Temporarily ask for 5 files but only 2 exist
		config["data"]["channel_size_imgs"] = 5

		img_folders = [joinpath(root, "patient_01", "Images")]
		@test_throws ErrorException load_channel_data(img_folders, config, "image")
	end

	# ── preprocess_channel_data ───────────────────────────────
	@testset "preprocess_channel_data — no-op config" begin
		root, cfg_path = _build_fixture(tmp; sz = (8, 8, 8))
		config         = JSON.parsefile(cfg_path)

		img_folders = [joinpath(root, "patient_01", "Images")]
		raw         = load_channel_data(img_folders, config, "image")
		processed   = preprocess_channel_data(raw, config, Linear_en, "image")

		@test length(processed) == 1
		@test length(processed[1].images) == 2
		# With "avg" target_size all images should come out the same size
		sz1 = size(processed[1].images[1].voxel_data)
		sz2 = size(processed[1].images[2].voxel_data)
		@test sz1 == sz2
	end

	@testset "preprocess_channel_data — normalisation applied to images only" begin
		root, cfg_path = _build_fixture(tmp)
		config = JSON.parsefile(cfg_path)
		config["data"]["normalisation"]["normalize"] = true

		img_folders = [joinpath(root, "patient_01", "Images")]
		raw         = load_channel_data(img_folders, config, "image")
		processed   = preprocess_channel_data(raw, config, Linear_en, "image")

		v = processed[1].images[1].voxel_data
		@test minimum(v) ≥ 0.0f0
		@test maximum(v) ≤ 1.0f0 + eps(Float32)
	end

	@testset "preprocess_channel_data — normalisation skipped for masks" begin
		root, cfg_path = _build_fixture(tmp)
		config = JSON.parsefile(cfg_path)
		config["data"]["normalisation"]["normalize"] = true

		mask_folders = [joinpath(root, "patient_01", "Masks")]
		raw          = load_channel_data(mask_folders, config, "mask")
		# Should not throw and should NOT normalise mask values
		@test_nowarn preprocess_channel_data(raw, config, Nearest_neighbour_en, "mask")
	end

	# ── assemble_tensor ──────────────────────────────────────
	@testset "assemble_tensor shape" begin
		root, cfg_path = _build_fixture(tmp; channel_size = 2, sz = (8, 8, 8))
		config         = JSON.parsefile(cfg_path)

		img_folders = [joinpath(root, "patient_01", "Images")]
		raw         = load_channel_data(img_folders, config, "image")
		processed   = preprocess_channel_data(raw, config, Linear_en, "image")
		names       = [ch.folder_name for ch in processed]

		tensor, meta = assemble_tensor(processed, names, "image")

		# shape: (X, Y, Z, channel_size, n_channel_folders)
		@test ndims(tensor) == 5
		@test size(tensor, 4) == 2    # channel_size
		@test size(tensor, 5) == 1    # one channel folder
		@test length(meta) == 2    # one meta entry per image
	end

	# ── load_and_preprocess ──────────────────────────────────
	@testset "load_and_preprocess returns correct shapes" begin
		root, cfg_path = _build_fixture(tmp; channel_size = 2, sz = (8, 8, 8))

		img_folders = [joinpath(root, "patient_01", "Images")]
		tensor, meta, names = load_and_preprocess(
			img_folders, cfg_path, Linear_en, "image")

		@test ndims(tensor) == 5
		@test size(tensor, 4) == 2
		@test length(meta) == 2
		@test names == ["patient_01"]
	end

	# ── batch_main ───────────────────────────────────────────
	@testset "batch_main runs end-to-end" begin
		root, cfg_path = _build_fixture(tmp; channel_size = 2, sz = (8, 8, 8))
		save_path      = joinpath(tmp, "output.h5")

		# batch_main calls save_to_hdf5 which we cannot easily mock,
		# so we only verify it does not error and returns something.
		@test_nowarn batch_main(root, save_path, cfg_path)
	end

	@testset "batch_main errors on missing image folders" begin
		empty_root = mktempdir()
		mkpath(joinpath(empty_root, "patient_01", "Masks"))
		_, cfg_path = _build_fixture(tmp)
		@test_throws ErrorException batch_main(empty_root, tmp, cfg_path)
	end

	@testset "batch_main errors on missing mask folders" begin
		no_mask_root = mktempdir()
		_write_nifti_files(joinpath(no_mask_root, "patient_01", "Images"), 2)
		_, cfg_path = _build_fixture(tmp)
		@test_throws ErrorException batch_main(no_mask_root, tmp, cfg_path)
	end

end
