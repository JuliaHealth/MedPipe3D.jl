"""
`proc_hdf5(h5, config, rng)`

Creates train/validation/test splits from an open HDF5 file and a config dict.

Priority:
1) If `config["learning"]["Train_Val_Test_JSON"]` is set, use it to match groups.
2) Else, if metadata contains `data_split`, use it.
3) Else, split by ratios in `config["learning"]["test_train_validation"]`.
"""
function proc_hdf5(h5::HDF5.File, config::Dict, rng::AbstractRNG)
    # Collect top-level groups (patients)
    group_names = [name for name in keys(h5) if isa(h5[name], HDF5.Group)]
    if isempty(group_names)
        error("No patient groups found in HDF5 file.")
    end

    # Deterministic order unless shuffled later
    sort!(group_names)

    # Try JSON-based split if provided
    split_json = config["learning"]["Train_Val_Test_JSON"]
    if split_json != false
        split_map = JSON.parsefile(split_json)
        return split_from_json(group_names, h5, split_map)
    end

    # Try metadata-based split if available
    split_from_meta = split_from_metadata(group_names, h5)
    if split_from_meta !== nothing
        return split_from_meta
    end

    # Fallback: ratio split
    return split_by_ratio(group_names, config, rng)
end

"""
`k_fold_split(indices, n_folds, fold)`

Splits indices into train/validation for a given fold.
"""
function k_fold_split(indices::Vector, n_folds::Int, fold::Int, rng::AbstractRNG)
    n = length(indices)
    if n_folds <= 1 || n == 0
        return indices, String[]
    end
    fold = clamp(fold, 1, n_folds)

    # Distribute remainder across the first folds
    base = div(n, n_folds)
    remn = rem(n, n_folds)
    fold_sizes = [base + (i <= remn ? 1 : 0) for i in 1:n_folds]

    start_idx = 1 + sum(fold_sizes[1:fold-1])
    end_idx = start_idx + fold_sizes[fold] - 1

    val = indices[start_idx:end_idx]
    train = vcat(indices[1:start_idx-1], indices[end_idx+1:end])
    return train, val
end

function split_from_json(group_names::Vector{String}, h5::HDF5.File, split_map::Dict)
    train = String[]
    val = String[]
    test = String[]

    # Normalize keys to String
    split_keys = Dict{String, Any}()
    for (k, v) in split_map
        key = lowercase(string(k))
        if key in ("val", "valid", "validation")
            split_keys["validation"] = v
        elseif key == "train"
            split_keys["train"] = v
        elseif key == "test"
            split_keys["test"] = v
        else
            split_keys[key] = v
        end
    end

    for group in group_names
        split = match_group_to_split(group, h5, split_keys)
        if split == "train"
            push!(train, group)
        elseif split == "validation"
            push!(val, group)
        elseif split == "test"
            push!(test, group)
        else
            # Default to train if no match
            push!(train, group)
        end
    end

    return Dict("train" => train, "validation" => val, "test" => test)
end

function split_from_metadata(group_names::Vector{String}, h5::HDF5.File)
    train = String[]
    val = String[]
    test = String[]
    found_any = false

    for group in group_names
        split = read_data_split(group, h5)
        if split === nothing
            continue
        end
        found_any = true
        if split == "train"
            push!(train, group)
        elseif split == "validation" || split == "val"
            push!(val, group)
        elseif split == "test"
            push!(test, group)
        else
            # Unknown split value: keep in train
            push!(train, group)
        end
    end

    if found_any
        return Dict("train" => train, "validation" => val, "test" => test)
    else
        return nothing
    end
end

function split_by_ratio(group_names::Vector{String}, config::Dict, rng::AbstractRNG)
    shuffle_groups = get(config["learning"], "shuffle", false)
    groups = copy(group_names)
    if shuffle_groups
        groups = shuffle(rng, groups)
    end

    ratios = parse_split_ratios(config["learning"]["test_train_validation"])
    ratio_sum = sum(ratios)
    if ratio_sum <= 0
        error("Invalid split ratios: sum must be > 0.")
    end
    if abs(ratio_sum - 1.0) > 1e-6
        ratios = ratios ./ ratio_sum
    end

    n = length(groups)
    n_train = floor(Int, ratios[1] * n)
    n_val = floor(Int, ratios[2] * n)
    n_test = n - n_train - n_val
    if n_test < 0
        n_test = 0
        n_val = max(n - n_train, 0)
    end

    train = groups[1:n_train]
    val = groups[n_train+1 : n_train+n_val]
    test = groups[n_train+n_val+1 : end]

    return Dict("train" => train, "validation" => val, "test" => test)
end

function parse_split_ratios(value)
    if isa(value, Tuple) || isa(value, NTuple)
        return collect(Float64, value)
    elseif isa(value, AbstractVector)
        return [Float64(v) for v in value]
    elseif isa(value, AbstractString)
        tuple_vals = string_to_tuple(value)
        return [Float64(v) for v in tuple_vals]
    else
        error("Unsupported split ratio format: $(typeof(value))")
    end
end

function match_group_to_split(group::String, h5::HDF5.File, split_keys::Dict{String, Any})
    file_path = read_file_path(group, h5)

    # Expected keys: train/validation/test
    for split_name in ("train", "validation", "test")
        if !haskey(split_keys, split_name)
            continue
        end
        patterns = split_keys[split_name]
        if !isa(patterns, AbstractVector)
            patterns = [patterns]
        end
        for p in patterns
            pat = string(p)
            if occursin(pat, group) || (file_path !== nothing && occursin(pat, file_path))
                return split_name
            end
        end
    end
    return nothing
end

function read_file_path(group::String, h5::HDF5.File)
    # Try images metadata first, then masks
    img_meta_path = group * "/images/metadata"
    mask_meta_path = group * "/masks/metadata"

    if group_exists(h5, img_meta_path)
        meta_group = h5[img_meta_path]
        if !isempty(keys(meta_group))
            first_meta_name = first(keys(meta_group))
            meta_sub = meta_group[first_meta_name]
            return safe_read_attribute(meta_sub, "file_path")
        end
    elseif group_exists(h5, mask_meta_path)
        meta_group = h5[mask_meta_path]
        if !isempty(keys(meta_group))
            first_meta_name = first(keys(meta_group))
            meta_sub = meta_group[first_meta_name]
            return safe_read_attribute(meta_sub, "file_path")
        end
    end
    return nothing
end

function read_data_split(group::String, h5::HDF5.File)
    img_meta_path = group * "/images/metadata"
    mask_meta_path = group * "/masks/metadata"

    if group_exists(h5, img_meta_path)
        meta_group = h5[img_meta_path]
        if !isempty(keys(meta_group))
            first_meta_name = first(keys(meta_group))
            meta_sub = meta_group[first_meta_name]
            return safe_read_attribute(meta_sub, "data_split")
        end
    elseif group_exists(h5, mask_meta_path)
        meta_group = h5[mask_meta_path]
        if !isempty(keys(meta_group))
            first_meta_name = first(keys(meta_group))
            meta_sub = meta_group[first_meta_name]
            return safe_read_attribute(meta_sub, "data_split")
        end
    end
    return nothing
end
