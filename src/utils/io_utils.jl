using HDF5
using JSON
using MedImages

# ────────────────────────────────────────────────────────────
# HDF5 low-level helpers
# ────────────────────────────────────────────────────────────

"""Return true if `path` names an existing HDF5 Group inside `h5`."""
function group_exists(h5::Union{HDF5.File, HDF5.Group}, path::AbstractString)::Bool
    try
        return isa(h5[path], HDF5.Group)
    catch
        return false
    end
end

"""
Read attribute `name` from `obj`, returning `default` when absent or on error.
Avoids crashes from missing metadata fields.
"""
function safe_read_attribute(obj, name::AbstractString; default=nothing)
    try
        haskey(attrs(obj), name) ? read_attribute(obj, name) : default
    catch
        default
    end
end

"""
Write every key/value pair in `meta` as an HDF5 attribute on `group`.
- Tuples / NTuples are collected to plain vectors (HDF5 requires array types).
- `Image_type` values are stringified.
- `nothing` values are silently skipped.
"""
function safe_write_meta(group::HDF5.Group, meta::Dict{String,Any})
    for (key, value) in meta
        value === nothing && continue
        if isa(value, Union{Tuple, NTuple})
            write_attribute(group, key, collect(value))
        elseif isa(value, Image_type)
            write_attribute(group, key, string(value))
        else
            write_attribute(group, key, value)
        end
    end
end

# ────────────────────────────────────────────────────────────
# HDF5 metadata reading
# ────────────────────────────────────────────────────────────

"""Read all HDF5 attributes from `meta_group` into a plain `Dict`."""
function read_metadata(meta_group)::Dict
    Dict(attr_name => read_attribute(meta_group, attr_name)
         for attr_name in keys(attrs(meta_group)))
end

# ────────────────────────────────────────────────────────────
# HDF5 batch loading
# ────────────────────────────────────────────────────────────

"""
Sort key for metadata channel indices.
Handles pure integers, strings ending in digits, and plain strings.
"""
function _meta_sort_key(k)
    if isa(k, AbstractString)
        n = tryparse(Int, k)
        n !== nothing && return (0, n)
        m = match(r"(\d+)$", k)
        m !== nothing && return (1, parse(Int, m.captures[1]))
        return (2, k)
    end
    return (3, string(k))
end

"""
Return one metadata Dict per channel from `meta_group`.
Handles: missing group → empty dicts; single shared entry → broadcast; partial list → fill with empty.
"""
function _metadata_per_channel(meta_group, num_channels::Int)::Vector{Dict}
    meta_group === nothing && return [Dict() for _ in 1:num_channels]
    meta_keys = sort!(collect(keys(meta_group)), by=_meta_sort_key)
    isempty(meta_keys) && return [Dict() for _ in 1:num_channels]

    metas = [read_metadata(meta_group[k]) for k in meta_keys]

    if length(metas) == num_channels
        return metas
    elseif length(metas) == 1
        return fill(metas[1], num_channels)
    else
        return [i <= length(metas) ? metas[i] : Dict() for i in 1:num_channels]
    end
end

"""
Parse a v2-format HDF5 data group (contains `data` dataset + optional `metadata` group).
Returns `(channel_data, channel_meta)` where each element corresponds to one image in the batch.
Supports 3-D (single image), 4-D (single image, multi-channel), and 5-D (batch × channel) arrays.
"""
function _process_data_group_v2(data_group::HDF5.Group)
    data = read(data_group["data"])
    nd   = ndims(data)

    nd ∈ (3, 4, 5) || error("Unsupported data dimensions: $nd")

    num_channels        = nd >= 4 ? size(data, 4) : 1
    num_images_in_batch = nd == 5 ? size(data, 5) : 1

    meta_group   = haskey(data_group, "metadata") ? data_group["metadata"] : nothing
    meta_by_chan = _metadata_per_channel(meta_group, num_channels)

    channel_data = Vector{Any}(undef, num_images_in_batch)
    channel_meta = Vector{Any}(undef, num_images_in_batch)

    if nd == 3
        channel_data[1] = [data]
        channel_meta[1] = meta_by_chan
    elseif nd == 4
        channel_data[1] = [@view data[:, :, :, j] for j in 1:num_channels]
        channel_meta[1] = meta_by_chan
    else
        for i in 1:num_images_in_batch
            channel_data[i] = [@view data[:, :, :, j, i] for j in 1:num_channels]
            channel_meta[i] = meta_by_chan
        end
    end

    return channel_data, channel_meta
end

"""
Parse a legacy-format HDF5 batch group (flat dataset keyed by `data_key`).
Returns `(channel_data, channel_meta)`.
"""
function process_data_group(batch_group, data_key::String, metadata_key::String)
    data                = read(batch_group[data_key])
    num_channels        = size(data, 4)
    num_images_in_batch = size(data, 5)

    channel_data = Vector{Any}()
    channel_meta = Vector{Any}()

    if !haskey(batch_group, metadata_key)
        @warn "No '$metadata_key' group found in batch group."
        return channel_data, channel_meta
    end

    meta_group   = batch_group[metadata_key]
    prefix       = data_key == "images" ? "image" : "mask"

    for i in 1:num_images_in_batch
        images_in_channel = Any[]
        meta_in_channel   = Any[]
        for j in 1:num_channels
            meta_key = "$(prefix)_$(j)"
            meta     = haskey(meta_group, meta_key) ? read_metadata(meta_group[meta_key]) : Dict()
            push!(meta_in_channel,   meta)
            push!(images_in_channel, data[:, :, :, j, i])
        end
        push!(channel_data, images_in_channel)
        push!(channel_meta, meta_in_channel)
    end

    return channel_data, channel_meta
end

"""
Load all image and mask batches (plus metadata) from an HDF5 file.

Returns `(image_batches, image_batch_metadata, mask_batches, mask_batch_metadata)`.
Each element of `image_batches` / `mask_batches` is a list of per-batch entries;
each entry is a list of channels; each channel is an array slice.
"""
function load_images_from_hdf5(hdf5_path::String)
    image_batches        = Any[]
    image_batch_metadata = Any[]
    mask_batches         = Any[]
    mask_batch_metadata  = Any[]

    h5open(hdf5_path, "r") do file
        for batch_name in keys(file)
            batch_group = file[batch_name]
            isa(batch_group, HDF5.Group) || continue

            for (key, batches, metas) in (
                    ("images", image_batches, image_batch_metadata),
                    ("masks",  mask_batches,  mask_batch_metadata))

                if !haskey(batch_group, key)
                    @warn "No '$key' found in '$batch_name'."
                    continue
                end

                obj = batch_group[key]
                if isa(obj, HDF5.Dataset)
                    data, meta = process_data_group(batch_group, key, "$(key)_metadata")
                elseif isa(obj, HDF5.Group) && haskey(obj, "data")
                    data, meta = _process_data_group_v2(obj)
                else
                    @warn "Unrecognised '$key' object in '$batch_name'."
                    continue
                end

                push!(batches, data)
                push!(metas,   meta)
            end
        end
    end

    return image_batches, image_batch_metadata, mask_batches, mask_batch_metadata
end

# ────────────────────────────────────────────────────────────
# HDF5 → MedImage / NIfTI conversion
# ────────────────────────────────────────────────────────────

"""
Reconstruct the original MedImage for `meta`, replace its voxel data with `data`,
and save it to `output_folder` using `suffix` appended before the file extension.
"""
function process_and_save_medimage(meta::Dict, data::AbstractArray,
                                   output_folder::String, suffix::String)
    original_path = meta["file_path"]

    original_image = if isdefined(MedImages, :load_images)
        MedImages.load_images(original_path)[1]
    elseif isdefined(MedImages, :load_image) &&
           hasmethod(MedImages.load_image, Tuple{String, String})
        img_type = get(meta, "image_type", "CT")
        MedImages.load_image(original_path, string(img_type))
    else
        MedImages.load_image(original_path)
    end

    updated = update_voxel_and_spatial_data(
        original_image, data,
        original_image.origin, original_image.spacing, original_image.direction
    )

    stem, ext   = splitext(basename(original_path))
    output_path = joinpath(output_folder, stem * suffix * ext)

    create_nii_from_medimage(updated, output_path)
    @info "Saved $(suffix) → $output_path"
end

"""
Convert every image and mask stored in `hdf5_path` back to NIfTI files in `output_folder`.
Appends `_image_after` / `_mask_after` to each filename.
"""
function convert_hdf5_to_medimages(hdf5_path::String, output_folder::String)
    image_batches, image_metas, mask_batches, mask_metas =
        load_images_from_hdf5(hdf5_path)

    for (batches, metas, suffix) in (
            (image_batches, image_metas, "_image_after"),
            (mask_batches,  mask_metas,  "_mask_after"))
        for (batch, batch_meta) in zip(batches, metas)
            for (channels, chan_meta) in zip(batch, batch_meta)
                for (arr, meta) in zip(channels, chan_meta)
                    process_and_save_medimage(meta, arr, output_folder, suffix)
                end
            end
        end
    end

    @info "All images and masks processed and saved to $output_folder."
end

# ────────────────────────────────────────────────────────────
# JSON split / class helpers
# ────────────────────────────────────────────────────────────

"""
Look up `channel_path` in a split or class JSON file.

- If `json_path` is `nothing` / `false`, returns `nothing`.
- If `class_names` is provided, prepends a numeric index to each class label
  (e.g. `"1_tumour"`), making ordering explicit.
"""
function get_class_or_split_from_json(channel_path::String,
                                      json_path,
                                      class_names=nothing)
    (json_path === nothing || json_path === false) && return nothing

    data = JSON.parsefile(json_path)

    if class_names !== nothing
        indexed = Dict(name => "$(i)_$name" for (i, name) in enumerate(class_names))
        for (cls, paths) in data
            any(p -> occursin(p, channel_path), paths) && return indexed[cls]
        end
    else
        for (key, paths) in data
            any(p -> occursin(p, channel_path), paths) && return key
        end
    end

    return nothing
end