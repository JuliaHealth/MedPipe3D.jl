using Test
  

@testset "parsing_utils" begin

    # ── string_to_tuple ───────────────────────────────────────
    @testset "string_to_tuple" begin
        @test string_to_tuple("(1.0, 2.0, 3.0)") === (1.0f0, 2.0f0, 3.0f0)
        @test string_to_tuple("1.0,2.0,3.0")     === (1.0f0, 2.0f0, 3.0f0)
        @test string_to_tuple("64,64,128")        === (64.0f0, 64.0f0, 128.0f0)
        # Leading/trailing whitespace inside parens
        @test string_to_tuple("( 1.0 , 1.0 , 2.5 )") === (1.0f0, 1.0f0, 2.5f0)
    end

    # ── parse_tuple3_float ────────────────────────────────────
    @testset "parse_tuple3_float" begin
        @test parse_tuple3_float("1.0,2.0,3.0")     === (1.0, 2.0, 3.0)
        @test parse_tuple3_float("(1.0, 2.0, 3.0)") === (1.0, 2.0, 3.0)
        @test_throws ErrorException parse_tuple3_float("1.0,2.0")        # too few
        @test_throws ErrorException parse_tuple3_float("1.0,2.0,3.0,4.0") # too many
    end

    # ── parse_tuple3_int ─────────────────────────────────────
    @testset "parse_tuple3_int" begin
        @test parse_tuple3_int("64,64,128")      === (64, 64, 128)
        @test parse_tuple3_int("(64, 64, 128)")  === (64, 64, 128)
        @test_throws ErrorException parse_tuple3_int("64,64")
    end

    # ── parse_optimizer_args ─────────────────────────────────
    @testset "parse_optimizer_args" begin
        # Floats: not valid Int, so stored as Float64
        d = parse_optimizer_args("lr=0.001,weight_decay=1e-5")
        @test typeof(d["lr"]) == Float64
        @test d["lr"]           ≈ 0.001
        @test d["weight_decay"] ≈ 1e-5

        # Whole number: Int tried first → stored as Int, not Float64
        d2 = parse_optimizer_args("epochs=50")
        @test typeof(d2["epochs"]) == Int
        @test d2["epochs"] === 50

        # String fallback
        d3 = parse_optimizer_args("mode=nesterov")
        @test typeof(d3["mode"]) == String
        @test d3["mode"] == "nesterov"

        # Malformed pair (no '=') is silently skipped
        d4 = parse_optimizer_args("lr=0.01,badentry,beta=0.9")
        @test  haskey(d4, "lr")
        @test  haskey(d4, "beta")
        @test !haskey(d4, "badentry")
    end

end