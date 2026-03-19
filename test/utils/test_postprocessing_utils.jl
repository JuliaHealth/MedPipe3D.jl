using Test
using CUDA

try
    using MedPipe3D
catch
    include(joinpath(@__DIR__, "..", "..", "src", "MedPipe3D.jl"))
    using .MedPipe3D
end

@testset "largest_connected_components (GPU post-processing)" begin

    if !CUDA.functional()
        @info "Skipping largest_connected_components tests: CUDA not functional on this system."
        return
    end

    # Simple 3×3×3 volume with two disjoint components of different sizes
    mask = zeros(Int32, 3, 3, 3)
    # Component 1: three voxels
    mask[1, 1, 1] = 1
    mask[1, 1, 2] = 1
    mask[1, 2, 1] = 1
    # Component 2: single voxel
    mask[3, 3, 3] = 1

    comps = largest_connected_components(mask, 1)

    @test length(comps) == 1
    largest = comps[1]
    @test largest[1, 1, 1] == 1
    @test largest[1, 1, 2] == 1
    @test largest[1, 2, 1] == 1
    @test largest[3, 3, 3] == 0
end

