"""
Performs inference using the given model and training state on the provided data.

# Arguments
- `tstate`: The current training state of the model.
- `model`: The model to be used for inference.
- `data`: The input data for which predictions are to be made.

# Returns
- `y_pred`: The predicted output from the model.
- `st`: The updated state after applying the model.
"""
function infer_model(tstate, model, data)
    y_pred, st = Lux.apply(model, CuArray(data), tstate.parameters, tstate.states)
    return y_pred, st
end



"""
Casts the given data to the appropriate device (CUDA or CPU) based on the configuration.

# Arguments
- `data`: The input data to be cast.
- `config`: A dictionary or struct containing configuration information, including the target device.

# Returns
- The data cast to the appropriate device.
"""
function cast_to_device(data, config)
    device = get(config, "device", "cpu")  # Default to CPU if device is not specified
    if device == "cuda"
        return CuArray(data)
        
    elseif device == "cpu"
        return data
    else
        error("Unknown device: $device")
    end
end