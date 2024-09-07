module validation

using utils
"""
Evaluates the model on the validation set and returns the evaluation metric.
# Arguments
- `val_indices`: Indices of the validation data.
- `model`: The model to be evaluated.
- `tstate`: The training state of the model.
- `config`: Configuration settings, including device preferences.
# Returns
- The evaluation metric computed from the model's predictions and the validation labels.
"""
function evaluate_validation(val_indices, model, tstate, config)
    all_metrics = []
    for val_index_sublist in val_indices
        data, label = fetch_and_preprocess_data(val_index_sublist)
        data = augment(data)
        if config.device == "CUDA"
            data = cu(data) # Move data to GPU
            label = cu(label) # Move data to GPU
        end
        y_pred, st = infer_model(tstate, model, data)
        metric = evaluate_metric(y_pred, label,config)
        push!(all_metrics, metric)
    end
    return all_metrics
end


end