# ────────────────────────────────────────────────────────────
# Tuple / numeric parsing
# ────────────────────────────────────────────────────────────

"""
	string_to_tuple(str) -> NTuple{N, Float32}

Parse a string like `"(1.0, 2.0, 3.0)"` or `"1.0,2.0,3.0"` into a Float32 tuple.
Parentheses and surrounding whitespace are stripped before splitting on commas.

# Examples
```julia
string_to_tuple("(1.0, 1.0, 2.5)")  # → (1.0f0, 1.0f0, 2.5f0)
string_to_tuple("64,64,128")        # → (64.0f0, 64.0f0, 128.0f0)
```
"""
function string_to_tuple(str::String)
	clean = replace(str, '(' => "", ')' => "")
	Tuple(parse(Float32, s) for s in split(clean, ','))
end

"""
	parse_tuple3_float(s) -> NTuple{3, Float64}

Strict 3-element Float64 variant.  Throws if the input does not contain
exactly three comma-separated values.
"""
function parse_tuple3_float(s::String)::NTuple{3, Float64}
	nums = parse.(Float64, split(strip(s, ['(', ')', ' ']), ','))
	length(nums) == 3 || error("Expected 3 values, got $(length(nums)): \"$s\"")
	(nums[1], nums[2], nums[3])
end

"""
	parse_tuple3_int(s) -> NTuple{3, Int}

Strict 3-element Int variant.
"""
function parse_tuple3_int(s::String)::NTuple{3, Int}
	nums = parse.(Int, split(strip(s, ['(', ')', ' ']), ','))
	length(nums) == 3 || error("Expected 3 values, got $(length(nums)): \"$s\"")
	(nums[1], nums[2], nums[3])
end

# ────────────────────────────────────────────────────────────
# Optimizer argument parsing  (new — from config refactor)
# ────────────────────────────────────────────────────────────

"""
	parse_optimizer_args(s) -> Dict{String, Any}

Parse a comma-separated `"key=value"` string into a typed Dict.

Values are coerced in order: `Int` → `Float64` → left as `String`.
Int is tried first so that whole numbers like `"50"` stay as `Int`
rather than being widened to `Float64`.

# Examples
```julia
parse_optimizer_args("lr=0.001,weight_decay=1e-5")
# → Dict("lr" => 0.001, "weight_decay" => 1.0e-5)

parse_optimizer_args("epochs=50")
# → Dict("epochs" => 50)          # Int, not 50.0

parse_optimizer_args("lr=0.001, betas=(0.9,0.999)")
# → Dict("lr" => 0.001, "betas" => "(0.9,0.999)")
```
"""
function parse_optimizer_args(s::String)::Dict{String, Any}
	d = Dict{String, Any}()
	for pair in split(s, ',')
		kv = split(strip(pair), '=')
		length(kv) == 2 || continue
        k = String(strip(kv[1]))
		v = String(strip(kv[2]))
		d[k] = something(tryparse(Int, v), tryparse(Float64, v), v)
	end
	d
end

# ────────────────────────────────────────────────────────────
# (Interactive helpers previously lived here; configuration.jl
# now owns all stdin / prompt logic so parsing_utils remains
# pure and side-effect free.)
# ────────────────────────────────────────────────────────────