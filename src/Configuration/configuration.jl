using JSON

# ────────────────────────────────────────────────────────────
# Typed parameter structs
# ────────────────────────────────────────────────────────────

struct ResamplingConfig
    strategy::String          # "avg" | "median" | "set" | "none"
    target_spacing::Union{NTuple{3,Float64}, Nothing}
    target_size::Union{NTuple{3,Int}, String}   # tuple or "avg"
end

struct NormalisationConfig
    standardize::Bool
    normalize::Bool
end

struct DataConfig
    batch_size::Int
    channel_size_imgs::Int
    channel_size_masks::Int
    resample_to_target::Bool
    resampling::ResamplingConfig
    normalisation::NormalisationConfig
end

struct AugmentationEntry
    name::String
    p_rand::Float64           # per-augmentation probability  
    params::Dict{String,Any}
end

struct AugmentationConfig
    order::Vector{String}
    processing_unit::String
    augmentations::Vector{AugmentationEntry}
end

struct SplitConfig
    json_path::Union{String, Nothing}   # explicit JSON with train/val/test lists
    ratios::NTuple{3,Float64}           # (train, val, test) — used when json_path is nothing
end

struct CrossValConfig
    enabled::Bool
    n_folds::Int
end

struct PatchConfig
    enabled::Bool
    size::Union{NTuple{3,Int}, Nothing}
    oversampling_probability::Float64
end

struct LearningConfig
    split::SplitConfig
    cross_val::CrossValConfig
    patch::PatchConfig
    invertible_augmentations::Bool
    shuffle::Bool
    metric::String
    largest_connected_component::Bool
    n_lcc::Int
    class_json_path::Union{String, Nothing}
    additional_json_paths::Vector{String}
end

struct EarlyStoppingConfig
    enabled::Bool
    patience::Int
    min_delta::Float64
    monitor::String
end

struct ModelConfig
    optimizer::String
    optimizer_args::Dict{String,Any}    # parsed key=value pairs
    num_epochs::Int
    loss::String
    early_stopping::EarlyStoppingConfig
end

struct PipelineConfig
    data::DataConfig
    augmentation::AugmentationConfig
    learning::LearningConfig
    model::ModelConfig
end

# ────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────

function _read(prompt::String, default)
    print("$prompt [default: $(repr(default))]: ")
    raw = strip(readline())
    isempty(raw) ? default : raw
end

_bool(s)  = parse(Bool, s)
_int(s)   = parse(Int, s)
_float(s) = parse(Float64, s)

function _tuple3_float(s::String)
    nums = parse.(Float64, split(strip(s, ['(',')',' ']), ','))
    length(nums) == 3 || error("Expected 3 values, got $(length(nums))")
    (nums[1], nums[2], nums[3])
end

function _tuple3_int(s::String)
    nums = parse.(Int, split(strip(s, ['(',')',' ']), ','))
    length(nums) == 3 || error("Expected 3 values, got $(length(nums))")
    (nums[1], nums[2], nums[3])
end

"""Parse "lr=0.001, beta1=0.9" into Dict("lr"=>0.001, "beta1"=>0.9)."""
function _parse_optimizer_args(s::String)::Dict{String,Any}
    d = Dict{String,Any}()
    for pair in split(s, ',')
        kv = split(strip(pair), '=')
        length(kv) == 2 || continue
        k, v = strip(kv[1]), strip(kv[2])
        d[k] = something(tryparse(Float64, v), tryparse(Int, v), v)
    end
    d
end

# ────────────────────────────────────────────────────────────
# Section builders
# ────────────────────────────────────────────────────────────

function _build_data_config()::DataConfig
    println("\n── Data Parameters ──────────────────────────────────")

    channel_imgs  = _int(_read("Channel size for images",  "4"))
    channel_masks = _int(_read("Channel size for masks",   "4"))
    batch_size    = _int(_read("Batch size",               "4"))
    # NOTE: batch_complete removed — incomplete batches are simply dropped
    #       ( use a DataLoader with drop_last=true semantics)

    resample_to_target = _bool(_read("Resample to first image? (true/false)", "false"))

    strategy = _read("Resample spacing strategy (avg/median/set/none)", "avg")
    target_spacing = nothing
    if strategy == "set"
        raw = _read("Target spacing, e.g. 1.0,1.0,1.0", "1.0,1.0,1.0")
        target_spacing = _tuple3_float(raw)
    end

    raw_size = _read("Resample size (avg or e.g. 128,128,128)", "avg")
    target_size = raw_size == "avg" ? "avg" : _tuple3_int(raw_size)

    standardize = _bool(_read("Standardisation?", "false"))
    normalize   = _bool(_read("Normalisation?",   "false"))

    DataConfig(
        batch_size,
        channel_imgs,
        channel_masks,
        resample_to_target,
        ResamplingConfig(strategy, target_spacing, target_size),
        NormalisationConfig(standardize, normalize)
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

function _collect_aug_params(name::String)::Dict{String,Any}
    p = Dict{String,Any}()
    if name == "Brightness transform"
        p["value"] = _float(_read("  value",  "0.2"))
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
        raw = _read("  axes, e.g. 1,2,3", "1,2,3")
        p["axes"] = parse.(Int, split(raw, ','))

    elseif name == "Scale transform"
        p["scale_factor"]     = _float(_read("  scale factor",      "1.0"))
        p["interpolator_enum"] = _read("  interpolator enum", "Linear_en")

    elseif name == "Gaussian blur transform"
        p["sigma"]           = _float(_read("  sigma",       "1.0"))
        p["kernel_size"]     = _int(_read("  kernel size",   "5"))
        p["shape"]           = _read("  shape (2D/3D)",      "3D")
        p["processing_unit"] = _read("  processing unit (GPU/CPU)", "GPU")

    elseif name == "Simulate low-resolution transform"
        p["blur_sigma"]       = _float(_read("  blur sigma",        "1.0"))
        p["kernel_size"]      = _int(_read("  kernel size",         "5"))
        p["downsample_scale"] = _float(_read("  downsample scale",  "2.0"))

    elseif name == "Elastic deformation transform"
        p["strength"]         = _float(_read("  strength",           "1.0"))
        p["interpolator_enum"] = _read("  interpolator enum", "Linear_en")
    end
    p
end

function _build_augmentation_config()::AugmentationConfig
    println("\n── Augmentation Parameters ──────────────────────────")
    println("Available augmentations (enter numbers separated by commas, or leave blank):")
    for (i, a) in enumerate(AUGMENTATION_MENU)
        println("  $i. $a")
    end

    raw_sel = strip(readline())
    if isempty(raw_sel)
        return AugmentationConfig(String[], "GPU", AugmentationEntry[])
    end

    indices = parse.(Int, split(raw_sel, ','))
    processing_unit = _read("Global processing unit (GPU/CPU)", "GPU")

    entries = AugmentationEntry[]
    for idx in indices
        name = AUGMENTATION_MENU[idx]
        println("\nConfiguring: $name")
        # per-augmentation p_rand
        p_rand = _float(_read("  p_rand (probability of applying this augmentation)", "0.5"))
        params = _collect_aug_params(name)
        push!(entries, AugmentationEntry(name, p_rand, params))
    end

    AugmentationConfig([e.name for e in entries], processing_unit, entries)
end

function _build_learning_config()::LearningConfig
    println("\n── Learning / Pipeline Parameters ───────────────────")

    # Train / val / test split
    use_json = _bool(_read("Use a JSON file for train/val/test split?", "false"))
    split_cfg = if use_json
        path = _read("  Path to split JSON", "")
        SplitConfig(isempty(path) ? nothing : path, (0.6, 0.2, 0.2))
    else
        raw = _read("  Train/val/test ratios, e.g. 0.6,0.2,0.2", "0.6,0.2,0.2")
        nums = parse.(Float64, split(raw, ','))
        SplitConfig(nothing, (nums[1], nums[2], nums[3]))
    end

    # Cross-validation
    cv_enabled = _bool(_read("Use n-fold cross-validation?", "false"))
    n_folds    = cv_enabled ? _int(_read("  Number of folds", "5")) : 1
    cv = CrossValConfig(cv_enabled, n_folds)

    # Patch / oversampling
    patch_enabled = _bool(_read("Use probabilistic patch oversampling?", "false"))
    patch = if patch_enabled
        raw_sz = _read("  Patch size, e.g. 64,64,64", "64,64,64")
        sz = _tuple3_int(raw_sz)
        prob = _float(_read("  Oversampling probability (0=random, 1=always foreground)", "0.5"))
        PatchConfig(true, sz, prob)
    else
        PatchConfig(false, nothing, 0.0)
    end

    invertible   = _bool(_read("Invertible augmentations?", "false"))
    shuffle      = _bool(_read("Shuffle channels?", "false"))
    metric       = _read("Evaluation metric", "dice")
    use_lcc      = _bool(_read("Use largest connected component post-processing?", "false"))
    n_lcc        = use_lcc ? _int(_read("  Number of components", "1")) : 1

    class_json   = let r = _read("Path to class JSON (or leave blank)", "")
        isempty(r) ? nothing : r
    end

    add_jsons = String[]
    while true
        r = _read("Additional JSON path (leave blank to stop)", "")
        isempty(r) && break
        push!(add_jsons, r)
    end

    LearningConfig(split_cfg, cv, patch, invertible, shuffle,
                   metric, use_lcc, n_lcc, class_json, add_jsons)
end

function _build_model_config()::ModelConfig
    println("\n── Model Parameters ─────────────────────────────────")

    optimizer  = _read("Optimizer (e.g. Adam, SGD)", "Adam")
    raw_args   = _read("Optimizer args, e.g. lr=0.001,weight_decay=1e-5", "lr=0.001")
    opt_args   = _parse_optimizer_args(raw_args)
    num_epochs = _int(_read("Number of epochs", "50"))
    loss       = _read("Loss function (e.g. dice, bce, l1, Custom)", "dice")
    loss == "Custom" && println("  ℹ  Pass your custom loss directly to Main_loop.")

    es_enabled = _bool(_read("Use early stopping?", "false"))
    es = if es_enabled
        patience  = _int(_read("  Patience",    "5"))
        min_delta = _float(_read("  Min delta", "0.001"))
        monitor   = _read("  Monitor metric", "val_loss")
        EarlyStoppingConfig(true, patience, min_delta, monitor)
    else
        EarlyStoppingConfig(false, 5, 0.001, "val_loss")
    end

    ModelConfig(optimizer, opt_args, num_epochs, loss, es)
end

# ────────────────────────────────────────────────────────────
# Serialisation helpers  (structs → plain Dict for JSON)
# ────────────────────────────────────────────────────────────

function _to_dict(c::ResamplingConfig)
    Dict("strategy"       => c.strategy,
         "target_spacing" => c.target_spacing,
         "target_size"    => c.target_size == "avg" ? "avg" : collect(c.target_size))
end

function _to_dict(c::NormalisationConfig)
    Dict("standardize" => c.standardize, "normalize" => c.normalize)
end

function _to_dict(c::DataConfig)
    Dict("batch_size"         => c.batch_size,
         "channel_size_imgs"  => c.channel_size_imgs,
         "channel_size_masks" => c.channel_size_masks,
         "resample_to_target" => c.resample_to_target,
         "resampling"         => _to_dict(c.resampling),
         "normalisation"      => _to_dict(c.normalisation),
         "has_mask"           => true)
end

function _to_dict(e::AugmentationEntry)
    Dict("name"   => e.name,
         "p_rand" => e.p_rand,
         "params" => e.params)
end

function _to_dict(c::AugmentationConfig)
    Dict("order"           => c.order,
         "processing_unit" => c.processing_unit,
         "augmentations"   => [_to_dict(e) for e in c.augmentations])
end

function _to_dict(c::SplitConfig)
    Dict("json_path" => c.json_path,
         "ratios"    => collect(c.ratios))
end

function _to_dict(c::CrossValConfig)
    Dict("enabled" => c.enabled, "n_folds" => c.n_folds)
end

function _to_dict(c::PatchConfig)
    Dict("enabled"                => c.enabled,
         "size"                   => isnothing(c.size) ? nothing : collect(c.size),
         "oversampling_probability" => c.oversampling_probability)
end

function _to_dict(c::LearningConfig)
    Dict("split"                    => _to_dict(c.split),
         "cross_val"                => _to_dict(c.cross_val),
         "patch"                    => _to_dict(c.patch),
         "invertible_augmentations" => c.invertible_augmentations,
         "shuffle"                  => c.shuffle,
         "metric"                   => c.metric,
         "largest_connected_component" => c.largest_connected_component,
         "n_lcc"                    => c.n_lcc,
         "class_json_path"          => c.class_json_path,
         "additional_json_paths"    => c.additional_json_paths)
end

function _to_dict(c::EarlyStoppingConfig)
    Dict("enabled"   => c.enabled,
         "patience"  => c.patience,
         "min_delta" => c.min_delta,
         "monitor"   => c.monitor)
end

function _to_dict(c::ModelConfig)
    Dict("optimizer"       => c.optimizer,
         "optimizer_args"  => c.optimizer_args,
         "num_epochs"      => c.num_epochs,
         "loss"            => c.loss,
         "early_stopping"  => _to_dict(c.early_stopping))
end

function _to_dict(c::PipelineConfig)
    Dict("data"         => _to_dict(c.data),
         "augmentation" => _to_dict(c.augmentation),
         "learning"     => _to_dict(c.learning),
         "model"        => _to_dict(c.model))
end

# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

"""
    create_config(save_path, config_name="config.json") -> String

Interactively builds a `PipelineConfig`, saves it as pretty-printed JSON,
and returns the file path.

Changes vs. the original `create_config_extended`:
- Parameters are grouped into typed structs (DataConfig, AugmentationConfig, …).
- `batch_complete` is removed; incomplete batches are dropped implicitly by the
  DataLoader (rethought as per TODO #2).
- Each augmentation now carries its own `p_rand` field (TODO #3).
- Optimizer arguments are parsed into a structured Dict rather than a raw string.
"""
function create_config(save_path::String, config_name::String="config.json")::String
    cfg = PipelineConfig(
        _build_data_config(),
        _build_augmentation_config(),
        _build_learning_config(),
        _build_model_config(),
    )

    json_path = joinpath(save_path, config_name)
    open(json_path, "w") do f
        print(f, JSON.json(_to_dict(cfg), 4))
    end
    println("\nConfiguration saved to $json_path")
    return json_path
end

"""
    modify_config(config, action, path, value=nothing) -> Dict

Navigate `config` along `path` and apply `action` (`:add`, `:modify`, `:remove`).
Returns the (mutated) config on success, or the unchanged config with an error
message on failure.

No functional changes from the original — kept as-is since the struct is still
returned as a plain Dict after deserialisation.
"""
function modify_config(config::Dict{String,Any},
                       action::Symbol,
                       path::Vector{String},
                       value=nothing)::Dict{String,Any}

    current = config
    for i in 1:length(path)-1
        key = path[i]
        if !haskey(current, key)
            if action === :add
                current[key] = Dict{String,Any}()
            else
                @warn "Key $(join(path[1:i], ".")) does not exist; cannot $action."
                return config
            end
        end
        next = current[key]
        if !(next isa Dict{String,Any})
            @warn "Non-dictionary node at $(join(path[1:i], ".")); cannot descend."
            return config
        end
        current = next
    end

    key = path[end]
    full = join(path, ".")

    if action === :add
        if haskey(current, key)
            @warn "Key $full already exists; use :modify to overwrite."
        else
            current[key] = value
            @info "Added $full = $(repr(value))"
        end
    elseif action === :modify
        if haskey(current, key)
            current[key] = value
            @info "Modified $full = $(repr(value))"
        else
            @warn "Key $full not found; cannot modify."
        end
    elseif action === :remove
        if haskey(current, key)
            delete!(current, key)
            @info "Removed $full"
        else
            @warn "Key $full not found; cannot remove."
        end
    else
        error("Unknown action $(repr(action)). Valid: :add, :modify, :remove")
    end

    return config
end