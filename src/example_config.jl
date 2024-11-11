{
    "model": {
        "patience": null,
        "early_stopping_metric": null,
        "optimizer_name": "Adam",
        "loss_function_name": "l1",
        "early_stopping": false,
        "early_stopping_min_delta": null,
        "optimizer_args": "lr=0.001",
        "num_epochs": 50
    },
    "data": {
        "batch_complete": true,
        "resample_size": "avg",
        "resample_to_target": false,
        "resample_to_spacing": "avg",
        "batch_size": 4,
        "standardization": false,
        "target_spacing": null,
        "channel_size_imgs": 4,
        "channel_size_masks": 3,
        "normalization": false,
        "has_mask": true
    },
    "augmentation": {
        "augmentations": {},
        "p_rand": 0.5,
        "processing_unit": "GPU",
        "order": []
    },
    "learning": {
        "Train_Val_Test_JSON": false,
        "n_folds": 1,
        "n_lcc": null,
        "invertible_augmentations": false,
        "class_JSON_path": false,
        "additional_JSON_path": false,
        "largest_connected_component": false,
        "patch_size": null,
        "metric": "dice",
        "n_cross_val": false,
        "patch_probabilistic_oversampling": false,
        "test_train_validation": [
            0.6,
            0.2,
            0.2
        ],
        "oversampling_probability": null,
        "shuffle": false
    }
}
