using JSON

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

function _read(prompt::String, default)
    print("$prompt [default: $(repr(default))]: ")
    raw = String(strip(readline()))
    isempty(raw) ? string(default) : raw
end

_bool(s)  = parse(Bool, s)
_int(s)   = parse(Int, s)
_float(s) = parse(Float64, s)

function _tuple3_float(s::String)
    nums = parse.(Float64, split(strip(s, ['(', ')', ' ']), ','))
    length(nums) == 3 || error("Expected 3 values, got $(length(nums))")
    [nums[1], nums[2], nums[3]]
end

function _tuple3_int(s::String)
    nums = parse.(Int, split(strip(s, ['(', ')', ' ']), ','))
    length(nums) == 3 || error("Expected 3 values, got $(length(nums))")
    [nums[1], nums[2], nums[3]]
end

# ────────────────────────────────────────────────────────────
# Section builders  (each returns a plain Dict)
# ────────────────────────────────────────────────────────────

function _build_data_config()::Dict{String, Any}
    println("\n── Data Parameters ──────────────────────────────────")

    channel_imgs  = _int(_read("Channel size for images", "4"))
    channel_masks = _int(_read("Channel size for masks", "4"))
    batch_size    = _int(_read("Batch size", "4"))

    resample_to_target = _bool(_read("Resample to first image? (true/false)", "false"))

    strategy       = _read("Resample spacing strategy (avg/median/set/none)", "avg")
    target_spacing = nothing
    if strategy == "set"
        target_spacing = _tuple3_float(
            _read("Target spacing, e.g. 1.0,1.0,1.0", "1.0,1.0,1.0"),
        )
    end

    raw_size    = _read("Resample size (avg or e.g. 128,128,128)", "avg")
    target_size = raw_size == "avg" ? "avg" : _tuple3_int(raw_size)

    standardize = _bool(_read("Standardisation?", "false"))
    normalize   = _bool(_read("Normalisation?", "false"))

    return Dict{String, Any}(
        "batch_size"         => batch_size,
        "channel_size_imgs"  => channel_imgs,
        "channel_size_masks" => channel_masks,
        "resample_to_target" => resample_to_target,
        "resampling"         => Dict(
            "strategy"       => strategy,
            "target_spacing" => target_spacing,
            "target_size"    => target_size,
        ),
        "normalisation"      => Dict(
            "standardize" => standardize,
            "normalize"   => normalize,
        ),
        "has_mask"           => true,
    )
end

const AUGMENTATION_MENU = [
    "Brightness transform",
    "Contrast augmentation transform",
    "Gamma transform",
    "Gaussian noise transform",
    "Rician noise transform",
    "Mirror transform",
    "Scale transform",
    "Gaussian blur transform",
    "Simulate low-resolution transform",
    "Elastic deformation transform",
]

function _collect_aug_params(name::String)::Dict{String, Any}
    p = Dict{String, Any}()
    if name == "Brightness transform"
        p["value"] = _float(_read("  value", "0.2"))
        p["mode"]  = _read("  mode (additive/multiplicative)", "additive")
    elseif name == "Contrast augmentation transform"
        p["factor"] = _float(_read("  contrast factor", "1.5"))
    elseif name == "Gamma transform"
        p["gamma"] = _float(_read("  gamma", "2.0"))
    elseif name == "Gaussian noise transform"
        p["variance"] = _float(_read("  variance", "0.01"))
    elseif name == "Rician noise transform"
        p["variance"] = _float(_read("  variance", "0.01"))
    elseif name == "Mirror transform"
        p["axes"] = parse.(Int, split(_read("  axes, e.g. 1,2,3", "1,2,3"), ','))
    elseif name == "Scale transform"
        p["scale_factor"]      = _float(_read("  scale factor", "1.0"))
        p["interpolator_enum"] = _read("  interpolator enum", "Linear_en")
    elseif name == "Gaussian blur transform"
        p["sigma"]           = _float(_read("  sigma", "1.0"))
        p["kernel_size"]     = _int(_read("  kernel size", "5"))
        p["shape"]           = _read("  shape (2D/3D)", "3D")
        p["processing_unit"] = _read("  processing unit (GPU/CPU)", "GPU")
    elseif name == "Simulate low-resolution transform"
        p["blur_sigma"]       = _float(_read("  blur sigma", "1.0"))
        p["kernel_size"]      = _int(_read("  kernel size", "5"))
        p["downsample_scale"] = _float(_read("  downsample scale", "2.0"))
    elseif name == "Elastic deformation transform"
        p["strength"]          = _float(_read("  strength", "1.0"))
        p["interpolator_enum"] = _read("  interpolator enum", "Linear_en")
    end
    return p
end

function _build_augmentation_config()::Dict{String, Any}
    println("\n── Augmentation Parameters ──────────────────────────")
    println("Available augmentations (enter numbers separated by commas, or leave blank):")
    for (i, a) in enumerate(AUGMENTATION_MENU)
        println("  $i. $a")
    end

    raw_sel = strip(readline())
    if isempty(raw_sel)
        return Dict{String, Any}("order" => [], "processing_unit" => "GPU", "augmentations" => [])
    end

    indices         = parse.(Int, split(raw_sel, ','))
    processing_unit = _read("Global processing unit (GPU/CPU)", "GPU")

    augmentations = []
    for idx in indices
        name = AUGMENTATION_MENU[idx]
        println("\nConfiguring: $name")
        p_rand = _float(_read("  p_rand (probability of applying)", "0.5"))
        push!(
            augmentations,
            Dict{String, Any}(
                "name"   => name,
                "p_rand" => p_rand,
                "params" => _collect_aug_params(name),
            ),
        )
    end

    return Dict{String, Any}(
        "order"           => [a["name"] for a in augmentations],
        "processing_unit" => processing_unit,
        "augmentations"   => augmentations,
    )
end

function _build_learning_config()::Dict{String, Any}
    println("\n── Learning / Pipeline Parameters ───────────────────")

    use_json   = _bool(_read("Use a JSON file for train/val/test split?", "false"))
    split_path = nothing
    ratios     = [0.6, 0.2, 0.2]
    if use_json
        r = _read("  Path to split JSON", "")
        split_path = isempty(r) ? nothing : r
    else
        raw    = _read("  Train/val/test ratios, e.g. 0.6,0.2,0.2", "0.6,0.2,0.2")
        ratios = parse.(Float64, split(raw, ','))
    end

    cv_enabled = _bool(_read("Use n-fold cross-validation?", "false"))
    n_folds    = cv_enabled ? _int(_read("  Number of folds", "5")) : 1

    patch_enabled = _bool(_read("Use probabilistic patch oversampling?", "false"))
    patch_size    = nothing
    patch_prob    = 0.0
    if patch_enabled
        patch_size = _tuple3_int(_read("  Patch size, e.g. 64,64,64", "64,64,64"))
        patch_prob = _float(
            _read("  Oversampling probability (0=random, 1=always foreground)", "0.5"),
        )
    end

    invertible = _bool(_read("Invertible augmentations?", "false"))
    shuffle    = _bool(_read("Shuffle channels?", "false"))
    metric     = _read("Evaluation metric", "dice")
    use_lcc    = _bool(_read("Use largest connected component post-processing?", "false"))
    n_lcc      = use_lcc ? _int(_read("  Number of components", "1")) : 1

    class_json = let r = _read("Path to class JSON (or leave blank)", "")
        isempty(r) ? nothing : r
    end

    add_jsons = String[]
    while true
        r = _read("Additional JSON path (leave blank to stop)", "")
        isempty(r) && break
        push!(add_jsons, r)
    end

    return Dict{String, Any}(
        "split" => Dict("json_path" => split_path, "ratios" => ratios),
        "cross_val" => Dict("enabled" => cv_enabled, "n_folds" => n_folds),
        "patch" => Dict(
            "enabled" => patch_enabled,
            "size" => patch_size,
            "oversampling_probability" => patch_prob,
        ),
        "invertible_augmentations"    => invertible,
        "shuffle"                     => shuffle,
        "metric"                      => metric,
        "largest_connected_component" => use_lcc,
        "n_lcc"                       => n_lcc,
        "class_json_path"             => class_json,
        "additional_json_paths"       => add_jsons,
    )
end

function _build_model_config()::Dict{String, Any}
    println("\n── Model Parameters ─────────────────────────────────")

    optimizer  = _read("Optimizer (e.g. Adam, SGD)", "Adam")
    opt_args   = parse_optimizer_args(
        _read("Optimizer args, e.g. lr=0.001,weight_decay=1e-5", "lr=0.001"),
    )
    num_epochs = _int(_read("Number of epochs", "50"))
    loss       = _read("Loss function (e.g. dice, bce, l1, Custom)", "dice")
    loss == "Custom" && println("  ℹ  Pass your custom loss directly to Main_loop.")

    es_enabled = _bool(_read("Use early stopping?", "false"))
    early_stopping = if es_enabled
        Dict{String, Any}(
            "enabled"   => true,
            "patience"  => _int(_read("  Patience", "5")),
            "min_delta" => _float(_read("  Min delta", "0.001")),
            "monitor"   => _read("  Monitor metric", "val_loss"),
        )
    else
        Dict{String, Any}(
            "enabled" => false,
            "patience" => 5,
            "min_delta" => 0.001,
            "monitor" => "val_loss",
        )
    end

    return Dict{String, Any}(
        "optimizer"      => optimizer,
        "optimizer_args" => opt_args,
        "num_epochs"     => num_epochs,
        "loss"           => loss,
        "early_stopping" => early_stopping,
    )
end

# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

"""
    create_config(save_path, config_name="config.json") -> String

Interactively build a pipeline config, save it as pretty-printed JSON,
and return the full file path.
"""
function create_config(save_path::String, config_name::String = "config.json")::String
    config = Dict{String, Any}(
        "data"         => _build_data_config(),
        "augmentation" => _build_augmentation_config(),
        "learning"     => _build_learning_config(),
        "model"        => _build_model_config(),
    )

    json_path = joinpath(save_path, config_name)
    open(json_path, "w") do f
        print(f, JSON.json(config, 4))
    end
    println("\nConfiguration saved to $json_path")
    return json_path
end

"""
    modify_config(config, action, path, value=nothing) -> Dict

Navigate `config` along `path` (vector of string keys) and apply `action`:
- `:add`    — insert a new key (warns if already present)
- `:modify` — update an existing key (warns if absent)
- `:remove` — delete a key (warns if absent)
"""
function modify_config(
    config::Dict{String, Any},
    action::Symbol,
    path::Vector{String},
    value = nothing,
)::Dict{String, Any}

    current = config
    for i in 1:(length(path)-1)
        key = path[i]
        if !haskey(current, key)
            action === :add ||
                (@warn "Key $(join(path[1:i], ".")) does not exist."; return config)
            current[key] = Dict{String, Any}()
        end
        next = current[key]
        isa(next, Dict{String, Any}) ||
            (@warn "Non-dict at $(join(path[1:i], "."))"; return config)
        current = next
    end

    key  = path[end]
    full = join(path, ".")

    if action === :add
        haskey(current, key) ?
        @warn("$full already exists; use :modify to overwrite.") :
        (@info("Added $full = $(repr(value))"); current[key] = value)
    elseif action === :modify
        haskey(current, key) ?
        (@info("Modified $full = $(repr(value))"); current[key] = value) :
        @warn("$full not found; cannot modify.")
    elseif action === :remove
        haskey(current, key) ?
        (@info("Removed $full"); delete!(current, key)) :
        @warn("$full not found; cannot remove.")
    else
        error("Unknown action $(repr(action)). Valid: :add, :modify, :remove")
    end

    return config
end
