
"""Holds the raw loaded images and their original metadata for one channel folder."""
struct ChannelData
    images::Vector{MedImage}
    metadata::Vector{Dict{String, Any}}
    folder_name::String
end

# ────────────────────────────────────────────────────────────
# batch_main
# ────────────────────────────────────────────────────────────

"""
	batch_main(main_folder, save_path, config_path=nothing, config_name="config.json")

Orchestrates loading, pre-processing, and saving of medical image/mask data.

# Arguments
- `main_folder`  : Root directory containing one sub-folder per patient.
- `save_path`    : Destination path for the output HDF5 file.
- `config_path`  : Path to an existing JSON config.  If `nothing`, a new one is
				   created interactively under `save_path`.
- `config_name`  : File name used when a new config is written.

# Description
For each patient folder the function searches for sub-directories whose names
contain `"image"` or `"mask"` (case-insensitive).  Both image and mask channels
are processed through the same pipeline in a single unified loop — they are only
dispatched to different interpolators and intensity-processing rules downstream.
"""
function batch_main(
    main_folder::String,
    save_path::String,
    config_path::Union{String, Nothing} = nothing,
    config_name::String = "config.json",
)
    # ── Config ────────────────────────────────────────────────
    if config_path === nothing
        config_path = create_config(save_path, config_name)
    end

    image_channel_folders = String[]
    mask_channel_folders  = String[]

    for patient_folder in readdir(main_folder)
        patient_path = joinpath(main_folder, patient_folder)
        isdir(patient_path) || continue
        for subfolder in readdir(patient_path)
            subfolder_path = joinpath(patient_path, subfolder)
            isdir(subfolder_path) || continue
            lc = lowercase(subfolder)
            if contains(lc, "image")
                push!(image_channel_folders, subfolder_path)
            elseif contains(lc, "mask")
                push!(mask_channel_folders, subfolder_path)
            end
        end
    end

    isempty(image_channel_folders) && error("No 'image' sub-folders found under $main_folder.")
    isempty(mask_channel_folders) && error("No 'mask' sub-folders found under $main_folder.")

    # Images use linear interpolation; masks use nearest-neighbour to preserve labels.
    channel_specs = [
        (image_channel_folders, Linear_en, "image"),
        (mask_channel_folders, Nearest_neighbour_en, "mask"),
    ]

    results       = Dict{String, Any}()
    channel_names = nothing

    for (folders, interpolator, kind) in channel_specs
        println("\nProcessing $kind channels…")
        tensor, metadata, names = load_and_preprocess(folders, config_path, interpolator, kind)
        results[kind] = (tensor, metadata)
        kind == "image" && (channel_names = names)
    end

    return save_to_hdf5(
        results["image"][1],
        results["image"][2],
        results["mask"][1],
        results["mask"][2],
        save_path,
        channel_names,
    )
end

# ────────────────────────────────────────────────────────────
# Stage 1 — Loading
# ────────────────────────────────────────────────────────────

"""
    load_channel_data(channel_paths, config, channel_type) -> Vector{ChannelData}

Load raw MedImages from disk and collect original metadata.
No spatial transformation is applied here.

# Arguments
- `channel_paths` : Folders, one per channel.
- `config`        : Parsed JSON config dict.
- `channel_type`  : `"image"` or `"mask"` — controls which `channel_size` key is read.

Returns one `ChannelData` per folder.  Folders that do not contain enough files
(and where the pipeline is not configured to pad) are collected as errors and
reported together before raising.
"""
function load_channel_data(
    channel_paths::Vector{String},
    config::Dict,
    channel_type::String,
)::Vector{ChannelData}

    channel_size = if channel_type == "image"
        config["data"]["channel_size_imgs"]
    elseif channel_type == "mask"
        config["data"]["channel_size_masks"]
    else
        error("channel_type must be \"image\" or \"mask\", got \"$channel_type\".")
    end

    dataset_splits = let v = config["learning"]["split"]["json_path"]
        v === nothing || v == false ? nothing : v
    end
    class_mapping = let v = config["learning"]["class_json_path"]
        v === nothing || v == false ? nothing : v
    end

    loaded = ChannelData[]
    errors = String[]

    for channel_path in channel_paths
        folder_name = basename(dirname(channel_path))
        println("  Loading channel: $folder_name")

        all_files = sort(
            filter(f -> isfile(joinpath(channel_path, f)), readdir(channel_path)),
        )
        image_files = [joinpath(channel_path, f) for f in all_files]

        if length(image_files) < channel_size
            # Insufficient files and no padding → defer error, continue to surface all problems at once.
            push!(
                errors,
                "Channel '$folder_name': expected $channel_size $channel_type files, found $(length(image_files)).",
            )
            continue
        end

        # Trim to channel_size (already sorted)
        needed_files = image_files[1:channel_size]
        images       = [load_images(fp)[1] for fp in needed_files]

        metadata = [
            Dict{String, Any}(
                "file_path"       => fp,
                "data_split"      => get_class_or_split_from_json(channel_path, dataset_splits),
                "class"           => get_class_or_split_from_json(channel_path, class_mapping),
                "patient_uid_org" => img.patient_uid,
                "shape_org"       => size(img.voxel_data),
                "spacing_org"     => img.spacing,
                "origin_org"      => img.origin,
                "direction_org"   => img.direction,
                "type_org"        => img.image_type,
            ) for (fp, img) in zip(needed_files, images)
        ]

        push!(loaded, ChannelData(images, metadata, folder_name))
    end

    if !isempty(errors)
        foreach(e -> @error(e), errors)
        error("Insufficient $channel_type files in one or more channels. See errors above.")
    end

    return loaded
end

# ────────────────────────────────────────────────────────────
# Stage 2 — Pre-processing
# ────────────────────────────────────────────────────────────

"""
	preprocess_channel_data(channels, config, interpolator, channel_type) -> Vector{ChannelData}

Apply all spatial and intensity transformations to already-loaded channel data.

Steps (in order):
1. Resample each channel to its own first image (if `resample_to_target`).
2. Resample all channels to a common spacing (`avg` / `median` / `set` / `none`).
3. Normalise / standardise intensity (images only, never masks).
4. Crop or pad to a common spatial size (`avg` or explicit tuple).
5. Validate that every image has the same final size.
"""
function preprocess_channel_data(
	channels::Vector{ChannelData},
	config::Dict,
	interpolator::Interpolator_enum,
	channel_type::String,
)::Vector{ChannelData}

	data_cfg           = config["data"]
	resample_to_target = data_cfg["resample_to_target"]
	spacing_strategy   = data_cfg["resampling"]["strategy"]
	target_spacing_cfg = data_cfg["resampling"]["target_spacing"]
	resample_size_cfg  = data_cfg["resampling"]["target_size"]
	do_normalize       = data_cfg["normalisation"]["normalize"]
	do_standardize     = data_cfg["normalisation"]["standardize"]

	# Convenience: rebuild mutable image lists so we can reassign
	imgs_per_channel = [copy(ch.images) for ch in channels]

	# ── Step 1: resample to first image in channel ─────────────
	if resample_to_target
		println("  Resampling each $channel_type channel to its own reference image.")
		imgs_per_channel = map(imgs_per_channel) do imgs
			ref = imgs[1]
			[resample_to_image(ref, img, interpolator) for img in imgs]
		end
	end

	# ── Step 2: common spacing ─────────────────────────────────
	if spacing_strategy == "set"
		target_sp = Tuple(Float32(s) for s in target_spacing_cfg)
		println("  Resampling all $channel_type files to target spacing: $target_sp")
		imgs_per_channel = map(imgs -> [resample_to_spacing(img, target_sp, interpolator) for img in imgs],
			imgs_per_channel)

	elseif spacing_strategy == "avg"
		all_sp = [img.spacing for imgs in imgs_per_channel for img in imgs]
		avg_sp = Tuple(Float32(mean(s)) for s in zip(all_sp...))
		println("  Resampling all $channel_type files to average spacing: $avg_sp")
		imgs_per_channel = map(imgs -> [resample_to_spacing(img, avg_sp, interpolator) for img in imgs],
			imgs_per_channel)

	elseif spacing_strategy == "median"
		all_sp = [img.spacing for imgs in imgs_per_channel for img in imgs]
		med_sp = Tuple(Float32(median(getindex.(all_sp, i))) for i in 1:length(all_sp[1]))
		println("  Resampling all $channel_type files to median spacing: $med_sp")
		imgs_per_channel = map(imgs -> [resample_to_spacing(img, med_sp, interpolator) for img in imgs],
			imgs_per_channel)

	elseif spacing_strategy in ("none", nothing, false)
		println("  Skipping spacing resampling for $channel_type files.")
	else
		@warn "Unknown spacing strategy '$(spacing_strategy)'; skipping."
	end

	# ── Step 3: intensity (images only) ───────────────────────
	if channel_type != "mask"
		if do_normalize
			println("  Normalising $channel_type files.")
			imgs_per_channel = map(imgs -> [normalize_image(img) for img in imgs], imgs_per_channel)
		end
		if do_standardize
			println("  Standardising $channel_type files.")
			imgs_per_channel = map(imgs -> [standardize_image(img) for img in imgs], imgs_per_channel)
		end
	end

	# ── Step 4: spatial size ───────────────────────────────────
	target_dim = if resample_size_cfg == "avg"
		all_sizes = [size(img.voxel_data) for imgs in imgs_per_channel for img in imgs]
		Tuple(Int(round(mean(getindex.(all_sizes, i)))) for i in 1:length(all_sizes[1]))
	else
		Tuple(Int(s) for s in resample_size_cfg)
	end
	println("  Resizing all $channel_type files to: $target_dim")
	imgs_per_channel = map(imgs -> [crop_or_pad(img, target_dim) for img in imgs],
		imgs_per_channel)

	# ── Step 5: size consistency check ────────────────────────
	expected = size(imgs_per_channel[1][1].voxel_data)
	for (ci, imgs) in enumerate(imgs_per_channel), (ii, img) in enumerate(imgs)
		sz = size(img.voxel_data)
		sz == expected || error(
			"$channel_type size mismatch at channel $ci, image $ii: " *
			"expected $expected, got $sz.")
	end

	# Rebuild ChannelData structs with updated images
	return [ChannelData(imgs_per_channel[i], channels[i].metadata, channels[i].folder_name)
			for i in eachindex(channels)]
end

# ────────────────────────────────────────────────────────────
# Stage 3 — Metadata finalisation + tensor assembly
# ────────────────────────────────────────────────────────────

"""
	assemble_tensor(channels, channel_names, channel_type)
		-> (final_tensor, flat_metadata)

Stamp final spatial metadata onto each image's metadata dict, concatenate all
voxel arrays into a single 5-D tensor `(X, Y, Z, channel, batch)`, and flatten
all metadata into one vector.
"""
function assemble_tensor(
	channels::Vector{ChannelData},
	channel_names::Vector{String},
	channel_type::String,
)
	for (i, ch) in enumerate(channels)
		for (j, img) in enumerate(ch.images)
			meta = ch.metadata[j]
			meta["name"] = channel_names[i] * "_$j"
			meta["shape_final"] = size(img.voxel_data)
			meta["spacing_final"] = img.spacing
			meta["origin_final"] = img.origin
			meta["direction_final"] = img.direction
		end
	end

	channel_tensors = map(enumerate(channels)) do (i, ch)
		println("  Stacking $channel_type channel $i along dim 4.")
		cat([img.voxel_data for img in ch.images]..., dims = 4)
	end

	println("  Concatenating all $channel_type channels into final 5-D tensor.")
	final_tensor = cat(channel_tensors..., dims = 5)
	flat_metadata = vcat([ch.metadata for ch in channels]...)

	@info "$(channel_type) tensor ready: $(size(final_tensor))"
	return final_tensor, flat_metadata
end

# ────────────────────────────────────────────────────────────
# Public combined entry point
# ────────────────────────────────────────────────────────────

"""
	load_and_preprocess(channel_paths, config_path, interpolator, channel_type)
		-> (final_tensor, metadata, channel_names)

Public façade that runs the two separated stages (loading → pre-processing)
and then assembles the final tensor.

Replaces the original monolithic `load_create_dataset_and_metadata`.
"""
function load_and_preprocess(
	channel_paths::Vector{String},
	config_path::String,
	interpolator::Interpolator_enum,
	channel_type::String,
)
	config = JSON.parsefile(config_path)
	channel_names = [basename(dirname(p)) for p in channel_paths]

	# Stage 1
	raw = load_channel_data(channel_paths, config, channel_type)

	# Stage 2
	processed = preprocess_channel_data(raw, config, interpolator, channel_type)

	# Stage 3
	final_tensor, metadata = assemble_tensor(processed, channel_names, channel_type)

	return final_tensor, metadata, channel_names
end
