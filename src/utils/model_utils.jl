using Lux
using MLDataDevices
using ADTypes

# ────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────

"""
	infer_model(tstate, model, data) -> (y_pred, st)

Run a single forward pass of `model` on `data`, automatically moving `data`
to the same device as `tstate.parameters`.

Returns the raw prediction tensor and the updated state `st`.
"""
function infer_model(tstate, model, data)
	dev   = MLDataDevices.get_device(tstate.parameters)
	input = data |> dev

	# 1. Put the model states into inference mode
	test_states = Lux.testmode(tstate.states)

	# 2. Apply the model using the test states
	y_pred, st = Lux.apply(model, input, tstate.parameters, test_states)

	return y_pred, st
end

# ────────────────────────────────────────────────────────────
# Tensor inspection
# ────────────────────────────────────────────────────────────

"""
	is_binary_tensor(tensor) -> Bool

Return `true` if every element of `tensor` is exactly `0` or `1`.
"""
function is_binary_tensor(tensor)::Bool
	return all(x -> x == 0 || x == 1, tensor)
end

"""
	check_if_binary_and_report(tensor)

Print every `(linear_index, value)` pair and the set of unique element types
found in `tensor`. Intended for debugging predicted segmentation masks.
"""
function check_if_binary_and_report(tensor)
	non_binary_values = [(i, v) for (i, v) in enumerate(tensor)]
	unique_types      = unique(typeof(v) for v in tensor)
	println("non_binary_values : ", non_binary_values)
	println("unique_types      : ", unique_types)
end

"""
Training utilities for Lux 0.5/1.x.

Defines:
- `single_train_step!` compatible with current code.
"""

function single_train_step!(::ADTypes.AutoZygote, loss_function, data, tstate)
	x, y = data

	# Move data to the same device as the parameters/states
	dev = MLDataDevices.get_device(tstate.parameters)
	x = x |> dev
	y = y |> dev

	ps = tstate.parameters
	st = tstate.states

	# Ensure parameters are on the same device as data (in case they were moved externally)
	ps = ps |> dev
	st = st |> dev

	obj_fn = function (model, ps_, st_, data_)
		x_, y_ = data_
		y_pred, st_new = Lux.apply(model, x_, ps_, st_)
		loss = loss_function(y_pred, y_)
		return loss, st_new, (; y_pred = y_pred)
	end

	_, loss, stats, new_tstate = Lux.Training.single_train_step!(ADTypes.AutoZygote(), obj_fn, (x, y), tstate)
	y_pred = get(stats, :y_pred, nothing)
	return y_pred, loss, stats, new_tstate
end

function single_train_step!(ad::ADTypes.AbstractADType, loss_function, data, tstate)
	if ad isa ADTypes.AutoZygote
		return single_train_step!(ADTypes.AutoZygote(), loss_function, data, tstate)
	end
	error("Unsupported AD backend: $(typeof(ad)). Only AutoZygote is supported in this pipeline.")
end
