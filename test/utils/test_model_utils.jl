using Test


@testset "model_utils" begin

	# ── is_binary_tensor ─────────────────────────────────────
	@testset "is_binary_tensor" begin
		@test is_binary_tensor([0, 1, 0, 1, 1])
		@test is_binary_tensor([0.0f0, 1.0f0, 0.0f0])
		@test !is_binary_tensor([0, 1, 2])
		@test !is_binary_tensor([0.0f0, 0.5f0, 1.0f0])
		@test is_binary_tensor(Int[])          # empty is vacuously true
	end

	# ── check_if_binary_and_report ───────────────────────────
	@testset "check_if_binary_and_report" begin
		# Should not throw; just prints
		@test_nowarn check_if_binary_and_report([0, 1, 2])
		@test_nowarn check_if_binary_and_report(Float32[])
	end

	# ── infer_model ───────────────────────────────────────────
	# We test infer_model with a minimal Lux model so the test
	# stays self-contained and does not require a real checkpoint.
	@testset "infer_model" begin
		using Lux, Random, MLDataDevices

		model  = Lux.Dense(4 => 2)
		rng    = Random.default_rng()
		ps, st = Lux.setup(rng, model)

		# Build a minimal training state stand-in
		# (Lux.Experimental.TrainState or a NamedTuple both work here)
		tstate = (parameters = ps, states = st)

		input = rand(Float32, 4, 3)   # 4 features, batch of 3
		y_pred, st2 = infer_model(tstate, model, input)

		@test size(y_pred) == (2, 3)
		@test y_pred isa AbstractArray{Float32}
	end

end
