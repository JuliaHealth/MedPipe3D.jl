#TODO: BIG DEVELOPMENTS in channel handling
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
Processes each validation path, predicts labels, computes metrics and loss, and aggregates them to provide overall validation performance.
"""

function evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes)
    # Initialize dictionaries to store metrics and losses by class
    val_metrics = Dict(i => Float64[] for i in 0:num_classes-1)
    val_losses = []
    println("Validating paths: $group_paths_val")
    for path in group_paths_val
        data, labels, _ = fetch_and_preprocess_data([path], h5, config)
        y_pred, _ = infer_model(tstate, model, data)
        val_loss = loss_function(y_pred, labels)
        metrics = evaluate_metric(y_pred, labels, config["learning"]["metric"])
        # Aggregate metrics and losses
        for (class_idx, metric_value) in metrics
            push!(val_metrics[class_idx], metric_value)
        end
        push!(val_losses, val_loss)
        println("Validation metric for $path: $metrics, Validation loss: $val_loss")
    end
    println("Validation metrics: $val_metrics, Validation losses: $val_losses")
    # Calculate mean metric and loss for each class
    mean_val_metrics = Dict{Int, Float64}()
    println(keys(val_metrics))
    # Process each class index
    for class_idx in keys(val_metrics)
        class_metrics = val_metrics[class_idx]
        class_mean = isempty(class_metrics) ? 0 : mean(class_metrics)
        mean_val_metrics[class_idx] = class_mean
    end
    mean_val_metrics = mean(values(mean_val_metrics))
    mean_val_loss = mean(val_losses)  # Ensuring NaN values do not affect the mean loss calculation

    println("Mean Validation Metrics by Class: $mean_val_metrics, Mean Validation Loss: $mean_val_loss")
    return mean_val_metrics, mean_val_loss
end

"""
`evaluate_metric(y_pred, labels, metric_type, threshold=0.5)`

Calculates evaluation metrics for predictions against true labels based on the specified metric type.

# Arguments
- `y_pred`: Predicted labels.
- `labels`: True labels.
- `metric_type`: Type of metric to calculate (e.g., 'dice', 'hausdorff').
- `threshold`: Threshold for converting probabilities to binary labels, if applicable.

# Returns
- A dictionary of results for each class.

# Description
Converts predictions and true labels to binary format and calculates the specified metric. Currently supports 'dice' and 'hausdorff' metrics.
"""
function evaluate_metric(y_pred, labels, metric_type, threshold=0.5)
    # Convert to GPU arrays if not already
    y_pred_gpu = y_pred isa CuArray ? y_pred : CuArray(y_pred)
    labels_gpu = labels isa CuArray ? labels : CuArray(labels)
    results = Dict{Int, Float64}()

    if metric_type == "dice"
        y_prob = 1 ./ (1 .+ exp.(-y_pred_gpu))
        eps_t = convert(eltype(y_prob), 1f-6)
        n_classes = size(y_prob, 4)

        if size(labels_gpu) == size(y_prob)
            for c in 1:n_classes
                yp = view(y_prob, :, :, :, c, :)
                yt = view(labels_gpu, :, :, :, c, :)
                inter = sum(yp .* yt)
                denom = sum(yp) + sum(yt)
                dice = (2 * inter + eps_t) / (denom + eps_t)
                results[c - 1] = Float64(dice)
            end
        elseif ndims(labels_gpu) == ndims(y_prob) && size(labels_gpu, 4) == 1
            y_true_slice = if eltype(labels_gpu) <: Integer
                view(labels_gpu, :, :, :, 1, :)
            else
                y_true_int = round.(Int, labels_gpu)
                view(y_true_int, :, :, :, 1, :)
            end
            for c in 1:n_classes
                yp = view(y_prob, :, :, :, c, :)
                mask = (y_true_slice .== (c - 1))
                inter = sum(yp .* mask)
                denom = sum(yp) + sum(mask)
                dice = (2 * inter + eps_t) / (denom + eps_t)
                results[c - 1] = Float64(dice)
            end
        else
            error("evaluate_metric expects labels to be one-hot or single-channel class indices.")
        end
    elseif metric_type == "hausdorff"
        n_classes = size(y_pred_gpu, 4)
        if size(labels_gpu) == size(y_pred_gpu)
            for c in 1:n_classes
                yp = view(y_pred_gpu, :, :, :, c, :)
                yt = view(labels_gpu, :, :, :, c, :)
                binary_pred = yp .>= threshold
                binary_labels = yt .>= threshold
                result = hausdorff_metric(binary_pred, binary_labels)
                results[c - 1] = Float64(result)
            end
        elseif ndims(labels_gpu) == ndims(y_pred_gpu) && size(labels_gpu, 4) == 1
            y_true_slice = if eltype(labels_gpu) <: Integer
                view(labels_gpu, :, :, :, 1, :)
            else
                y_true_int = round.(Int, labels_gpu)
                view(y_true_int, :, :, :, 1, :)
            end
            for c in 1:n_classes
                yp = view(y_pred_gpu, :, :, :, c, :)
                binary_pred = yp .>= threshold
                binary_labels = (y_true_slice .== (c - 1))
                result = hausdorff_metric(binary_pred, binary_labels)
                results[c - 1] = Float64(result)
            end
        else
            error("evaluate_metric expects labels to be one-hot or single-channel class indices.")
        end
    else
        throw(ArgumentError("Unsupported metric: $metric_type"))
    end

    return results
end

#TODO: rethink if it is not dice
function calculate_accuracy(y_pred, y_true)
    predictions = argmax(y_pred, dims=4)  # Zwraca indeksy maksymalnych wartości wzdłuż wymiaru klas
    true_classes = argmax(y_true, dims=4)  # Analogicznie dla prawdziwych etykiet

    correct = sum(predictions .== true_classes)  # Liczymy, gdzie przewidziana klasa zgadza się z prawdziwą
    total = length(predictions)  # Całkowita liczba pikseli

    return correct / total  # Procent poprawnych predykcji
end
