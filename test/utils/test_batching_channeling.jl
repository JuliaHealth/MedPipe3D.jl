using Test
using MedPipe3D

@testset "batching & channeling tensor fill" begin

    @testset "happy path fills tensor with reshaped images" begin
        n_batches  = 2
        n_channels = 3
        X, Y, Z    = 4, 5, 6

        tensor = zeros(Float32, n_batches, n_channels, X, Y, Z)

        # Build flat list of simple ramp volumes so we can check placement
        input_images = []
        for b in 1:n_batches, c in 1:n_channels
            img = medimage_from_array(fill(Float32((b - 1) * n_channels + c), X, Y, Z))
            push!(input_images, img)
        end

        fill_batch_tensor!(tensor, input_images, n_batches, n_channels)

        @test size(tensor) == (n_batches, n_channels, X, Y, Z)
        @test all(tensor[1, 1, :, :, :] .== 1f0)
        @test all(tensor[1, 2, :, :, :] .== 2f0)
        @test all(tensor[2, 3, :, :, :] .== 6f0)
    end

    @testset "mismatched input image count throws" begin
        n_batches  = 2
        n_channels = 2
        tensor = zeros(Float32, n_batches, n_channels, 2, 2, 2)

        img = medimage_from_array(ones(Float32, 2, 2, 2))
        input_images = [img]  # too few

        @test_throws ErrorException fill_batch_tensor!(tensor, input_images, n_batches, n_channels)
    end

    @testset "spatial size mismatch throws" begin
        n_batches  = 1
        n_channels = 1
        tensor = zeros(Float32, n_batches, n_channels, 4, 4, 4)

        # 3×3×3 instead of 4×4×4
        img = medimage_from_array(ones(Float32, 3, 3, 3))
        input_images = [img]

        @test_throws ErrorException fill_batch_tensor!(tensor, input_images, n_batches, n_channels)
    end

end

