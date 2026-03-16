using Test
using MedImages
using Statistics: mean, std

# ── Helpers ───────────────────────────────────────────────────

"""Build a minimal MedImage with synthetic voxel data."""
function _make_image(sz::Tuple, fill_val::Float32=1.0f0;
                     spacing=(1.0, 1.0, 1.0))
    arr = fill(fill_val, sz)
    return medimage_from_array(arr; spacing=spacing)
end

function _make_rand_image(sz::Tuple; spacing=(1.0, 1.0, 1.0))
    arr = rand(Float32, sz)
    return medimage_from_array(arr; spacing=spacing)
end

# ── Tests ────────────────────────────────────────────────────

@testset "image_utils" begin

    # ── medimage_from_array ───────────────────────────────────
    @testset "medimage_from_array" begin
        arr = rand(Float32, 8, 8, 8)
        img = medimage_from_array(arr)
        @test size(img.voxel_data) == (8, 8, 8)
        @test img.spacing          == (1.0, 1.0, 1.0)
        @test img.origin           == (0.0, 0.0, 0.0)
        @test img.patient_id       == "unknown"

        # Custom spacing is stored
        img2 = medimage_from_array(arr; spacing=(2.0, 2.0, 3.0), patient_id="p001")
        @test img2.spacing    == (2.0, 2.0, 3.0)
        @test img2.patient_id == "p001"
    end

    # ── normalize_image ───────────────────────────────────────
    @testset "normalize_image" begin
        img      = _make_rand_image((10, 10, 10))
        normed   = normalize_image(img)
        v        = normed.voxel_data
        @test minimum(v) ≥ 0.0f0
        @test maximum(v) ≤ 1.0f0 + eps(Float32)
        # Spatial metadata is preserved (spacing and origin stay the same)
        @test normed.spacing == img.spacing
        @test normed.origin  == img.origin

        # Flat image (all same value) should not throw
        flat = _make_image((4, 4, 4), 5.0f0)
        @test_nowarn normalize_image(flat)
    end

    # ── standardize_image ─────────────────────────────────────
    @testset "standardize_image" begin
        img     = _make_rand_image((10, 10, 10))
        std_img = standardize_image(img)
        v       = std_img.voxel_data
        @test abs(mean(v)) < 1e-5          # mean ≈ 0
        @test abs(std(v) - 1.0f0) < 1e-4  # std  ≈ 1
        # Spatial metadata preserved
        @test std_img.spacing == img.spacing
        @test std_img.origin  == img.origin

        # Flat image should not throw
        flat = _make_image((4, 4, 4), 3.0f0)
        @test_nowarn standardize_image(flat)
    end

    # ── crop_or_pad ───────────────────────────────────────────
    @testset "crop_or_pad" begin
        # No-op when sizes already match
        img = _make_image((8, 8, 8))
        @test size(crop_or_pad(img, (8, 8, 8)).voxel_data) == (8, 8, 8)

        # Pure crop: source clearly larger than target on all axes
        big     = _make_rand_image((16, 16, 16))
        cropped = crop_or_pad(big, (8, 8, 8))
        @test size(cropped.voxel_data) == (8, 8, 8)

        # Pure pad: source clearly smaller than target on all axes
        small  = _make_rand_image((4, 4, 4))
        padded = crop_or_pad(small, (8, 8, 8))
        @test size(padded.voxel_data) == (8, 8, 8)

        # Mixed: crop axis 1 (12→8), pad axis 2 (4→8), keep axis 3 (8→8)
        # crop_beg axis 1: floor((12-8)/2)+1 = 3, crop_size = min(12-3+1, 8) = 8 → view 3:10 ✓
        # axes 2 & 3 not cropped (size_diff ≤ 0) → crop_mi skipped for those axes
        # then pad_mi brings axis 2 from 4 to 8
        rect   = _make_rand_image((12, 4, 8))
        result = crop_or_pad(rect, (8, 8, 8))
        @test size(result.voxel_data) == (8, 8, 8)
    end

end