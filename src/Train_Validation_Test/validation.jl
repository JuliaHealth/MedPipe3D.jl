function evaluate_validation(group_paths_val, h5, model, tstate, loss_function, config, num_classes)
    # Initialize dictionaries to store metrics and losses by class
    val_metrics = Dict(i => Float64[] for i in 0:num_classes-1)
    val_losses = []
    println("Validating paths: $group_paths_val")
    for path in group_paths_val
        data, labels, _ = fetch_and_preprocess_data([path], h5, config)
        y_pred, _ = infer_model(tstate, model, data)
        _, val_loss, _, _ = Lux.Training.single_train_step!(Lux.Experimental.ADTypes.AutoZygote(), loss_function, (data, labels), tstate)
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


function evaluate_metric(y_pred, labels, metric_type, threshold=0.5)
    # Convert to GPU arrays if not already
    class_idx = maximum(labels)
    y_pred_gpu = CuArray(y_pred)
    labels_gpu = CuArray(labels)
    results = Dict()
    # Convert to binary masks for the current class
    for i in 0:size(labels, 4)
        binary_pred = (y_pred_gpu .== class_idx)
        binary_labels = (labels_gpu .== class_idx)
        # Calculate metrics per class
        if metric_type == "dice"
#TODO: to jest notatka do pod poprawkę channeli
            #result = dice_metric(y_pred(i 4), labels(i 4))
        elseif metric_type == "hausdorff"
            result = hausdorff_metric(binary_pred, binary_labels)
        else
            throw(ArgumentError("Unsupported metric: $metric_type"))
        end
    
        results[class_idx] = result
    return results
end


function calculate_accuracy(y_pred, y_true)
    predictions = argmax(y_pred, dims=4)  # Zwraca indeksy maksymalnych wartości wzdłuż wymiaru klas
    true_classes = argmax(y_true, dims=4)  # Analogicznie dla prawdziwych etykiet

    correct = sum(predictions .== true_classes)  # Liczymy, gdzie przewidziana klasa zgadza się z prawdziwą
    total = length(predictions)  # Całkowita liczba pikseli

    return correct / total  # Procent poprawnych predykcji
end