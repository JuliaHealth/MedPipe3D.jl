using image_processing, manage_indicies, get_metric, logging

"""
    evaluate_patches(test_data, tstates, test_time_augs, model, config)

Evaluates the model on patches of test data for given states and augmentations.

# Arguments
- `test_data`: The test data to evaluate.
- `tstates`: The states to use for testing.
- `test_time_augs`: The augmentations to apply during testing.
- `model`: The model to evaluate.
- `config`: Configuration settings, including logging preferences.

# Returns
- `results`: The predicted results for each patch.
- `test_metrics`: The evaluation metrics for each patch.
"""
function evaluate_patches(test_data, tstates, test_time_augs, model, config)
    results = []
    test_metrics = []
    for tstate_curr in tstates
        for aug in test_time_augs
            patch_results = []
            for patch in divide_to_patches(test_data)
                y_pred, st = main_loop.infer_model(tstate_curr, model, test_data)
                push!(patch_results, y_pred)
            end
            y_pred = merge_patches(patch_results)
        end
        y_pred = reverse_aug(y_pred)
        push!(results, y_pred)
        metr = evaluate_metric(y_pred, test_label,attributes, config)
        push!(test_metrics, metr)
        if config.log_metrics
            log_metric(config.logger, "test_metric", metr, epoch)
        end
    end
    return results, test_metrics
end

"""
    process_results(results, test_metrics, config)

Processes the results to obtain the largest connected components and computes the mean.

# Arguments
- `results`: The predicted results to process.
- `test_metrics`: The evaluation metrics to process.
- `config`: Configuration settings, including logging preferences.

# Returns
- `y_pred`: The mean of the processed results.
- `metr`: The mean of the test metrics.
"""
function process_results(results, test_metrics, config)
    for i in 1:length(results)
        results[i] = largest_connected_components(results[i] .> treshold)
    end
    y_pred = mean(results)
    metr = mean(test_metrics)

    return y_pred, metr
end

"""
    evaluate_test_set(test_indices, model, tstates, test_time_augs, config)

Evaluates the test set and returns all test metrics.

# Arguments
- `test_indices`: The indices of the test data to evaluate.
- `model`: The model to evaluate.
- `tstates`: The states to use for testing.
- `test_time_augs`: The augmentations to apply during testing.
- `config`: Configuration settings, including logging preferences.

# Returns
- `all_test_metrics`: The evaluation metrics for all test data.
"""
function evaluate_test_set(test_indices, model, tstates, test_time_augs, config,metadata_ref,logger)
    all_test_metrics = []
    for test_index in test_indices
        test_data, test_label,attributes = fetch_and_preprocess_data(test_index)
        results, test_metrics = evaluate_patches(test_data, tstates, test_time_augs, model, config,attributes)

        y_pred, metr = process_results(results, test_metrics, config)
        #TODO get output folder from config and get filename from metadata - like patiend id or original file name
        save_results(y_pred, filename, metadata_ref)
        push!(all_test_metrics, metr)
    end

    if config.log_metrics
        log_metric(logger, "mean_metric", metr, epoch)
    end

    return all_test_metrics
end

