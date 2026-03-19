using Test
using HDF5
using JSON
using MedImages




"""Thin shim so the private _metadata_per_channel is reachable from tests."""
_test_metadata_per_channel(meta_group, n) = _metadata_per_channel(meta_group, n)

# ── Fixtures ──────────────────────────────────────────────────

"""
Write a minimal v2-format HDF5 file with both images and masks groups.
Returns the file path.
"""
function _write_v2_hdf5(path::String; batch_size = 2, channels = 2, sz = (8, 8, 8))
    h5open(path, "w") do f
        for b in 1:batch_size
            bg = create_group(f, "batch_$b")

            for kind in ("images", "masks")
                kg = create_group(bg, kind)
                data = rand(Float32, sz..., channels, 1)
                kg["data"] = data
                mg = create_group(kg, "metadata")
                for c in 1:channels
                    cg = create_group(mg, "ch_$c")
                    write_attribute(cg, "spacing", [1.0, 1.0, 1.0])
                end
            end
        end
    end
    return path
end

"""Write a minimal class/split JSON and return its path."""
function _write_class_json(path::String)
    data = Dict(
        "tumour"  => ["/data/patient_01/images"],
        "healthy" => ["/data/patient_02/images"],
    )
    open(path, "w") do f
        print(f, JSON.json(data))
    end
    return path
end

# ── Tests ─────────────────────────────────────────────────────

@testset "io_utils" begin

    tmp = mktempdir()

    # ── group_exists ─────────────────────────────────────────
    @testset "group_exists" begin
        h5path = joinpath(tmp, "test_group.h5")
        h5open(h5path, "w") do f
            create_group(f, "present")
        end
        h5open(h5path, "r") do f
            @test group_exists(f, "present")
            @test !group_exists(f, "absent")
        end
    end

    # ── safe_read_attribute ───────────────────────────────────
    @testset "safe_read_attribute" begin
        h5path = joinpath(tmp, "test_attr.h5")
        h5open(h5path, "w") do f
            g = create_group(f, "g")
            write_attribute(g, "key", 42)
        end
        h5open(h5path, "r") do f
            g = f["g"]
            @test safe_read_attribute(g, "key") == 42
            @test safe_read_attribute(g, "missing") === nothing
            @test safe_read_attribute(g, "missing"; default = "fallback") == "fallback"
        end
    end

    # ── safe_write_meta ───────────────────────────────────────
    @testset "safe_write_meta" begin
        h5path = joinpath(tmp, "test_meta.h5")
        h5open(h5path, "w") do f
            g = create_group(f, "meta")
            meta = Dict{String, Any}(
                "int_val"   => 7,
                "float_val" => 3.14,
                "tuple_val" => (1.0, 2.0, 3.0),
                "skip_me"   => nothing,
            )
            safe_write_meta(g, meta)
        end
        h5open(h5path, "r") do f
            g = f["meta"]
            @test read_attribute(g, "int_val") == 7
            @test read_attribute(g, "float_val") ≈ 3.14
            @test read_attribute(g, "tuple_val") == [1.0, 2.0, 3.0]
            @test !haskey(attrs(g), "skip_me")
        end
    end

    # ── read_metadata ─────────────────────────────────────────
    @testset "read_metadata" begin
        h5path = joinpath(tmp, "test_readmeta.h5")
        h5open(h5path, "w") do f
            g = create_group(f, "m")
            write_attribute(g, "a", 1)
            write_attribute(g, "b", "hello")
        end
        h5open(h5path, "r") do f
            meta = read_metadata(f["m"])
            @test meta["a"] == 1
            @test meta["b"] == "hello"
        end
    end

    # ── _metadata_per_channel ─────────────────────────────────
    @testset "_metadata_per_channel" begin
        # nothing → n empty dicts
        result = _test_metadata_per_channel(nothing, 3)
        @test length(result) == 3
        @test all(isempty, result)
    end

    # ── load_images_from_hdf5 ─────────────────────────────────
    @testset "load_images_from_hdf5" begin
        h5path = _write_v2_hdf5(
            joinpath(tmp, "test_load.h5");
            batch_size = 2,
            channels = 2,
            sz = (8, 8, 8),
        )
        img_batches, img_meta, mask_batches, mask_meta = load_images_from_hdf5(h5path)

        # Two batch groups → two entries each
        @test length(img_batches) == 2
        @test length(mask_batches) == 2

        # Each batch entry: 1 image-in-batch (5-D data, size-5 = 1)
        @test length(img_batches[1]) == 1
        @test length(mask_batches[1]) == 1

        # Each image-in-batch has 2 channels
        @test length(img_batches[1][1]) == 2
        @test length(mask_batches[1][1]) == 2
    end

    # ── get_class_or_split_from_json ─────────────────────────
    @testset "get_class_or_split_from_json" begin
        json_path = _write_class_json(joinpath(tmp, "classes.json"))

        @test get_class_or_split_from_json("/data/patient_01/images/scan.nii", json_path) ==
              "tumour"
        @test get_class_or_split_from_json("/data/patient_02/images/scan.nii", json_path) ==
              "healthy"

        # No match → nothing
        @test get_class_or_split_from_json("/data/patient_99/images/scan.nii", json_path) ===
              nothing

        # json_path falsy → nothing
        @test get_class_or_split_from_json("/any/path", nothing) === nothing
        @test get_class_or_split_from_json("/any/path", false) === nothing

        # With class_names → indexed label ("tumour" is index 2 in the list)
        result = get_class_or_split_from_json(
            "/data/patient_01/images/scan.nii",
            json_path,
            ["healthy", "tumour"],
        )
        @test result == "2_tumour"
    end

end
