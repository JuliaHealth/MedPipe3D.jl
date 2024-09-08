using ComputerVisionMetrics

export evaluate_metric



"""
    evaluate_metric(output, label, attributes, config::Configuration)

Evaluate the specified metric between the given `output` and `label` based on the provided `config`.

# Arguments
- `output`: The predicted segmentation mask.
- `label`: The ground truth segmentation mask.
- `attributes`: Additional attributes that may contain class information.
- `config::Configuration`: Configuration object that specifies the metric to use and whether to use GPU.

# Returns
- The calculated metric value.

# Throws
- `ArgumentError` if an unsupported metric is specified in the configuration.

# Example
```julia
config = Configuration(metric=:dice, use_gpu=false)
dice_score = evaluate_metric(prediction, ground_truth, attributes, config)
```
"""
function evaluate_metric(output, label, attributes, config::Configuration)
    if isempty(label)
        # Extract class information from attributes
        classes = [attr.class for attr in attributes]
        # Apply argmax to the output to get predicted classes
        predicted_classes = argmax(output, dims=2)
        # Calculate accuracy
        accuracy = sum(predicted_classes .== classes) / length(classes)
        return accuracy
    else
        if config.metric == :hausdorff
            return hausdorff_metric(output, label)
        elseif config.metric == :dice
            return dice_metric(output, label)
        else
            throw(ArgumentError("Unsupported metric: $(config.metric)"))
        end
    end
end
