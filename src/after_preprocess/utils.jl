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
    y_pred, st = Lux.apply(model, CuArray(data), tstate_glob.parameters, tstate_glob.states)
    return y_pred, st
end
