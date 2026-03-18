using Test

try
    using MedPipe3D
catch
    include(joinpath(@__DIR__, "..", "..", "src", "MedPipe3D.jl"))
    using .MedPipe3D
end

@testset "training utilities" begin

    @testset "get_optimiser returns known optimisers" begin
        opt = get_optimiser("Adam")
        @test typeof(opt) <: Optimisers.Adam

        opt2 = get_optimiser("rmsprop")
        @test typeof(opt2) <: Optimisers.RMSProp
    end

    @testset "get_optimiser throws on unknown name" begin
        @test_throws ErrorException get_optimiser("not_an_optimizer")
    end

    @testset "get_loss_function returns known losses" begin
        l1 = get_loss_function("l1")
        @test typeof(l1) <: (Lux.GenericLossFunction)
        l2 = get_loss_function("mse")
        @test typeof(l2) <: (Lux.GenericLossFunction)
        l3 = get_loss_function("dice")
        @test l3 === dice_loss
    end

    @testset "get_loss_function throws on unknown name" begin
        @test_throws ErrorException get_loss_function("made_up_loss")
    end

    @testset "dice_loss basic properties" begin
        # Perfect overlap → loss ≈ 0
        y_true = ones(Float32, 4, 4, 4, 1, 1)
        y_pred = fill(10.0f0, size(y_true))  # sigmoid(10) ≈ 1
        @test dice_loss(y_pred, y_true) ≤ 1e-3

        # No overlap → loss close to 1
        y_true_zero = zeros(Float32, 4, 4, 4, 1, 1)
        y_pred_one  = fill(10.0f0, size(y_true_zero))
        @test dice_loss(y_pred_one, y_true_zero) ≈ 1f0 atol=1e-3
    end

end

