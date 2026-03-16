using Test
using JSON

# ── Helpers ───────────────────────────────────────────────────

"""
Run `f()` with stdin replaced by a temp file containing `answers`
(one answer per line). Uses a file instead of IOBuffer because
Julia 1.10's redirect_stdin does not accept IOBuffer directly.
"""
function with_stdin(f::Function, answers::Vector{String})
    tmp = tempname()
    write(tmp, join(answers, "\n") * "\n")
    open(tmp, "r") do io
        redirect_stdin(f, io)
    end
end

"""Return the minimal answer list that accepts all defaults."""
function _default_data_answers()
    # channel_imgs, channel_masks, batch_size,
    # resample_to_target, spacing_strategy,
    # resample_size, standardize, normalize
    return fill("", 8)
end

function _default_aug_answers()
    # augmentation selection (blank = none)
    return [""]
end

function _default_learning_answers()
    # use_json, ratios, cv, patch, invertible,
    # shuffle, metric, lcc, class_json, add_json (blank stops loop)
    return fill("", 10)
end

function _default_model_answers()
    # optimizer, opt_args, epochs, loss, early_stopping
    return fill("", 5)
end

function _all_default_answers()
    vcat(
        _default_data_answers(),
        _default_aug_answers(),
        _default_learning_answers(),
        _default_model_answers(),
    )
end

# ── Tests ─────────────────────────────────────────────────────

@testset "pipeline_config" begin

    tmp = mktempdir()

    # ── create_config writes valid JSON ──────────────────────
    @testset "create_config default" begin
        cfg_path = with_stdin(_all_default_answers()) do
            create_config(tmp, "test_config.json")
        end

        @test isfile(cfg_path)
        cfg = JSON.parsefile(cfg_path)

        # Top-level keys
        for k in ("data", "augmentation", "learning", "model")
            @test haskey(cfg, k)
        end

        # Data defaults
        @test cfg["data"]["batch_size"]        == 4
        @test cfg["data"]["channel_size_imgs"] == 4
        @test cfg["data"]["channel_size_masks"]== 4
        @test cfg["data"]["resample_to_target"]== false
        @test cfg["data"]["resampling"]["strategy"]     == "avg"
        @test cfg["data"]["resampling"]["target_size"]  == "avg"
        @test cfg["data"]["normalisation"]["normalize"] == false

        # Augmentation defaults (no augmentations selected)
        @test cfg["augmentation"]["augmentations"] == []
        @test cfg["augmentation"]["processing_unit"] == "GPU"

        # Learning defaults
        @test cfg["learning"]["cross_val"]["enabled"]  == false
        @test cfg["learning"]["cross_val"]["n_folds"]  == 1
        @test cfg["learning"]["patch"]["enabled"]      == false
        @test cfg["learning"]["metric"]                == "dice"

        # Model defaults
        @test cfg["model"]["optimizer"]   == "Adam"
        @test cfg["model"]["num_epochs"]  == 50
        @test cfg["model"]["loss"]        == "dice"
        @test cfg["model"]["optimizer_args"]["lr"] ≈ 0.001
        @test cfg["model"]["early_stopping"]["enabled"] == false
    end

    # ── create_config with non-default values ────────────────
    @testset "create_config custom values" begin
        answers = vcat(
            # data
            ["2",       # channel_imgs
             "1",       # channel_masks
             "8",       # batch_size
             "true",    # resample_to_target
             "set",     # spacing strategy
             "1.0,1.0,2.0",  # target_spacing (prompted because strategy==set)
             "64,64,128",    # resample_size
             "true",    # standardize
             "false"],  # normalize
            # augmentation: select entry 1 (Brightness transform)
            ["1",       # selected index
             "GPU",     # processing unit
             "0.3",     # p_rand
             "0.1",     # value
             "multiplicative"],  # mode
            # learning
            ["false",   # use_json
             "0.7,0.15,0.15",  # ratios
             "true",    # cv
             "3",       # n_folds
             "true",    # patch
             "32,32,32",# patch_size
             "0.7",     # oversampling_probability
             "false",   # invertible
             "false",   # shuffle
             "iou",     # metric
             "true",    # lcc
             "2",       # n_lcc
             "",        # class_json (blank)
             ""],       # additional_json (blank stops)
            # model
            ["SGD",
             "lr=0.01,momentum=0.9",
             "100",
             "bce",
             "true",    # early_stopping
             "10",      # patience
             "0.0001",  # min_delta
             "val_iou"],# monitor
        )

        cfg_path = with_stdin(answers) do
            create_config(tmp, "custom_config.json")
        end

        cfg = JSON.parsefile(cfg_path)

        @test cfg["data"]["batch_size"]                         == 8
        @test cfg["data"]["channel_size_imgs"]                  == 2
        @test cfg["data"]["resampling"]["strategy"]             == "set"
        @test cfg["data"]["resampling"]["target_spacing"]       == [1.0, 1.0, 2.0]
        @test cfg["data"]["resampling"]["target_size"]          == [64, 64, 128]
        @test cfg["data"]["normalisation"]["standardize"]       == true

        aug = cfg["augmentation"]["augmentations"][1]
        @test aug["name"]          == "Brightness transform"
        @test aug["p_rand"]        ≈  0.3
        @test aug["params"]["mode"]== "multiplicative"

        @test cfg["learning"]["cross_val"]["n_folds"]           == 3
        @test cfg["learning"]["patch"]["size"]                  == [32, 32, 32]
        @test cfg["learning"]["metric"]                         == "iou"
        @test cfg["learning"]["n_lcc"]                          == 2

        @test cfg["model"]["optimizer"]                         == "SGD"
        @test cfg["model"]["optimizer_args"]["momentum"]        ≈  0.9
        @test cfg["model"]["early_stopping"]["enabled"]         == true
        @test cfg["model"]["early_stopping"]["patience"]        == 10
        @test cfg["model"]["early_stopping"]["monitor"]         == "val_iou"
    end

    # ── modify_config ─────────────────────────────────────────
    @testset "modify_config :add" begin
        cfg = Dict{String,Any}("data" => Dict{String,Any}("batch_size" => 4))
        modify_config(cfg, :add, ["data", "new_key"], 99)
        @test cfg["data"]["new_key"] == 99

        # Adding to a non-existent nested path creates intermediate dicts
        modify_config(cfg, :add, ["data", "nested", "deep"], "hello")
        @test cfg["data"]["nested"]["deep"] == "hello"

        # Duplicate add is a no-op (warns, does not overwrite)
        modify_config(cfg, :add, ["data", "new_key"], 0)
        @test cfg["data"]["new_key"] == 99   # unchanged
    end

    @testset "modify_config :modify" begin
        cfg = Dict{String,Any}("model" => Dict{String,Any}("lr" => 0.001))
        modify_config(cfg, :modify, ["model", "lr"], 0.01)
        @test cfg["model"]["lr"] ≈ 0.01

        # Modifying a missing key is a no-op
        modify_config(cfg, :modify, ["model", "nonexistent"], 5)
        @test !haskey(cfg["model"], "nonexistent")
    end

    @testset "modify_config :remove" begin
        cfg = Dict{String,Any}("a" => Dict{String,Any}("b" => 1, "c" => 2))
        modify_config(cfg, :remove, ["a", "b"])
        @test !haskey(cfg["a"], "b")
        @test  haskey(cfg["a"], "c")

        # Removing a missing key is a no-op
        modify_config(cfg, :remove, ["a", "ghost"])
        @test  haskey(cfg["a"], "c")
    end

    @testset "modify_config :unknown raises" begin
        cfg = Dict{String,Any}()
        @test_throws ErrorException modify_config(cfg, :explode, ["x"])
    end

end