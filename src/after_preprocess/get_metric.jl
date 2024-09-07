module MetricEvaluator

using ComputerVisionMetrics

export evaluate_metric


"""
    evaluate_metric(image, label, config::Configuration)

Evaluate the specified metric between the given `image` and `label` based on the provided `config`.

# Arguments
- `image`: The predicted segmentation mask.
- `label`: The ground truth segmentation mask.
- `config::Configuration`: Configuration object that specifies the metric to use and whether to use GPU.

# Returns
- The calculated metric value.

# Throws
- `ArgumentError` if an unsupported metric is specified in the configuration.

# Example
```julia
config = Configuration(metric=:dice, use_gpu=false)
dice_score = evaluate_metric(prediction, ground_truth, config)
```
"""
function evaluate_metric(image, label, config::Configuration)
  
    if config.use_gpu
        image = CUDA.array(image)
        label = CUDA.array(label)
    end

    if config.metric == :hausdorff
        return hausdorff_metric(image, label)
    elseif config.metric == :dice
        return dice_metric(image, label)
    else
        throw(ArgumentError("Unsupported metric: $(config.metric)"))
    end
end

end