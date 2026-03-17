# Channel convention (consistent with batch_loader.jl, apply.jl, model_utils.jl):
#
#   dim 1-3 → spatial (W, H, D)
#   dim 4   → channels
#               • model INPUT:  input modalities (e.g. T1, T2, FLAIR)
#               • model OUTPUT: class logits / predictions per class
#   dim 5   → batch
#
# evaluate_metric and evaluate_validation operate exclusively on model OUTPUT tensors,
# so dim 4 always means n_classes here. This is confirmed by:
#   - size(y_prob, 4)     → n_classes in the dice branch
#   - size(y_pred_gpu, 4) → n_classes in the hausdorff branch
#   - argmax(..., dims=4) → class prediction in calculate_accuracy
#
# The dual meaning of dim 4 (modalities in / classes out) is an acknowledged design
# tension documented in apply.jl. It does not affect this file since validation only
# ever sees model output tensors, never raw input images.

"""
`evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes)`

Evaluates the model on validation data and calculates the average metrics and losses.

# Arguments
- `group_paths_val`: Paths to validation groups.
- `h5`: HDF5 file handle.
- `model`: Trained machine learning model.
- `tstate`: Training state with optimizer settings.
- `loss_function`: Function to calculate loss.
- `config`: Configuration settings.
- `num_classes`: Total number of classes including background.

# Returns
- Tuple of mean validation metrics and mean validation loss.

# Description
Processes each validation path, predicts labels, computes metrics and loss, and aggregates
them to provide overall validation performance. The metric type is read from
`config["learning"]["metric"]` and can be "dice", "hausdorff", or "accuracy".
"""
function evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes)
    val_metrics = Dict(i => Float64[] for i in 0:(num_classes - 1))
    val_losses  = Float32[]
    println("Validating paths: $group_paths_val")

    for path in group_paths_val
        data, labels, _ = fetch_and_preprocess_data([path], h5, config)
        y_pred, _       = infer_model(tstate, model, data)
        val_loss        = loss_function(y_pred, labels)
        metrics         = evaluate_metric(y_pred, labels, config["learning"]["metric"])

        for (class_idx, metric_value) in metrics
            push!(val_metrics[class_idx], metric_value)
        end
        push!(val_losses, val_loss)
        println("Validation metric for $path: $metrics, Validation loss: $val_loss")
    end

    println("Validation metrics: $val_metrics, Validation losses: $val_losses")

    mean_val_metrics_dict = Dict{Int, Float64}()
    for class_idx in keys(val_metrics)
        class_metrics = val_metrics[class_idx]
        mean_val_metrics_dict[class_idx] = isempty(class_metrics) ? 0.0 : mean(class_metrics)
    end

    overall_mean_metric = isempty(mean_val_metrics_dict) ? 0.0 : mean(values(mean_val_metrics_dict))
    mean_val_loss       = isempty(val_losses) ? 0.0f0 : mean(val_losses)

    println("Mean Validation Metrics by Class: $mean_val_metrics_dict, " *
            "Overall Mean Metric: $overall_mean_metric, " *
            "Mean Validation Loss: $mean_val_loss")

    return overall_mean_metric, mean_val_loss
end


"""
`evaluate_metric(y_pred, labels, metric_type, threshold=0.5)`

Calculates evaluation metrics for predictions against true labels.

# Arguments
- `y_pred`      : Predicted tensor [W, H, D, C_out, B] where dim 4 = class logits.
- `labels`      : Ground-truth tensor, either one-hot [W,H,D,C,B] or single-channel
                  class indices [W,H,D,1,B].
- `metric_type` : `"dice"`, `"hausdorff"`, or `"accuracy"`.
- `threshold`   : Binarisation threshold for hausdorff (default 0.5).

# Returns
- `Dict{Int, Float64}` mapping class index (0-based) to metric value.
"""
function evaluate_metric(y_pred, labels, metric_type, threshold = 0.5)
    y_pred_gpu = y_pred isa CuArray ? y_pred : CuArray(y_pred)
    labels_gpu = labels isa CuArray ? labels : CuArray(labels)
    results    = Dict{Int, Float64}()

    if metric_type == "dice"
        y_prob    = 1 ./ (1 .+ exp.(-y_pred_gpu))
        eps_t     = convert(eltype(y_prob), 1.0f-6)
        n_classes = size(y_prob, 4)

        if size(labels_gpu) == size(y_prob)
            for c in 1:n_classes
                yp    = view(y_prob,    :, :, :, c, :)
                yt    = view(labels_gpu, :, :, :, c, :)
                inter = sum(yp .* yt)
                denom = sum(yp) + sum(yt)
                results[c - 1] = Float64((2 * inter + eps_t) / (denom + eps_t))
            end
        elseif ndims(labels_gpu) == ndims(y_prob) && size(labels_gpu, 4) == 1
            y_true_int   = eltype(labels_gpu) <: Integer ?
                labels_gpu : round.(Int, labels_gpu)
            y_true_slice = view(y_true_int, :, :, :, 1, :)
            for c in 1:n_classes
                yp    = view(y_prob, :, :, :, c, :)
                mask  = (y_true_slice .== (c - 1))
                inter = sum(yp .* mask)
                denom = sum(yp) + sum(mask)
                results[c - 1] = Float64((2 * inter + eps_t) / (denom + eps_t))
            end
        else
            error("evaluate_metric: labels must be one-hot or single-channel class indices.")
        end

    elseif metric_type == "hausdorff"
        n_classes = size(y_pred_gpu, 4)

        if size(labels_gpu) == size(y_pred_gpu)
            for c in 1:n_classes
                yp             = view(y_pred_gpu, :, :, :, c, :)
                yt             = view(labels_gpu, :, :, :, c, :)
                binary_pred    = yp .>= threshold
                binary_labels  = yt .>= threshold
                results[c - 1] = Float64(hausdorff_metric(binary_pred, binary_labels))
            end
        elseif ndims(labels_gpu) == ndims(y_pred_gpu) && size(labels_gpu, 4) == 1
            y_true_int   = eltype(labels_gpu) <: Integer ?
                labels_gpu : round.(Int, labels_gpu)
            y_true_slice = view(y_true_int, :, :, :, 1, :)
            for c in 1:n_classes
                yp             = view(y_pred_gpu, :, :, :, c, :)
                binary_pred    = yp .>= threshold
                binary_labels  = (y_true_slice .== (c - 1))
                results[c - 1] = Float64(hausdorff_metric(binary_pred, binary_labels))
            end
        else
            error("evaluate_metric: labels must be one-hot or single-channel class indices.")
        end

    elseif metric_type == "accuracy"
        # Delegate to calculate_accuracy — returns a single overall accuracy value
        # stored under class key 0, since accuracy is not decomposed per class.
        results[0] = Float64(calculate_accuracy(y_pred_gpu, labels_gpu))

    else
        throw(ArgumentError("Unsupported metric: \"$metric_type\". " *
                            "Supported: \"dice\", \"hausdorff\", \"accuracy\"."))
    end

    return results
end


"""
`calculate_accuracy(y_pred, y_true)`

Computes voxel-level classification accuracy by comparing argmax predictions to
argmax ground-truth labels along the class dimension (dim 4).

# Arguments
- `y_pred` : Predicted tensor [W, H, D, C_out, B] — dim 4 holds class logits.
- `y_true` : Ground-truth tensor [W, H, D, C_out, B] — must be one-hot encoded.

# Returns
- `Float64` fraction of voxels where the predicted class equals the true class.

# Notes
- Accuracy is a coarse metric for segmentation — biased toward background in
  imbalanced datasets. Dice or Hausdorff are preferred for segmentation quality.
- This is now wired into `evaluate_metric` as metric_type `"accuracy"`, making it
  available anywhere in the pipeline that accepts a metric string from config.
- For single-channel integer label maps, convert to one-hot first, or use `"dice"`
  which natively handles both label formats.
"""
function calculate_accuracy(y_pred, y_true)
    predictions  = argmax(y_pred, dims = 4)
    true_classes = argmax(y_true, dims = 4)
    correct      = sum(predictions .== true_classes)
    total        = length(predictions)
    return correct / total
end