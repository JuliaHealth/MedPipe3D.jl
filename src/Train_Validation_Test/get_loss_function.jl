"""
`get_loss_function(name::String)`

A helper function to retrieve predefined loss functions based on their name, suitable for use in machine learning models.

# Arguments
- `name`: A string indicating the name of the desired loss function, e.g., "L1", "MSE", "CrossEntropy".

# Returns
- Returns the corresponding loss function object from the Lux library.

# Errors
- Throws an error if the provided loss function name does not match any of the predefined options.
"""
function get_loss_function(name::String)
    name = lowercase(name)
    if name == "l1" || name == "mae"
        return Lux.MAELoss()
    elseif name == "l2" || name == "mse"
        return Lux.MSELoss()
    elseif name == "crossentropy"
        return Lux.CrossEntropyLoss()
    elseif name == "dice"
        return dice_loss
    else
        error("Unsupported or unrecognized loss function: $name. You may need to define this loss function manually.")
    end
end

"""
`dice_loss(y_pred, y_true)`

Basic Dice loss for binary masks or one-hot encoded multi-class masks.
Applies a sigmoid to predictions to keep them in [0, 1].
"""
function dice_loss(y_pred, y_true; eps = 1f-6)
    y_prob = 1 ./ (1 .+ exp.(-y_pred))
    y_true_f = float.(y_true)
    eps_t = convert(eltype(y_prob), eps)

    # If there is a channel dimension, compute mean Dice across channels
    if ndims(y_prob) >= 4 && size(y_prob, 4) > 1
        sum_loss = zero(eltype(y_prob))
        n_classes = size(y_prob, 4)
        if size(y_true_f) == size(y_prob)
            for c in 1:n_classes
                yp = view(y_prob, :, :, :, c, :)
                yt = view(y_true_f, :, :, :, c, :)
                inter = sum(yp .* yt)
                denom = sum(yp) + sum(yt)
                sum_loss += 1 - (2 * inter + eps_t) / (denom + eps_t)
            end
        elseif ndims(y_true_f) == ndims(y_prob) && size(y_true_f, 4) == 1
            # Single-channel labels: compute per-class dice without in-place one-hot
            y_true_int = round.(Int, y_true_f)
            y_true_slice = view(y_true_int, :, :, :, 1, :)
            for c in 1:n_classes
                yp = view(y_prob, :, :, :, c, :)
                mask = (y_true_slice .== (c - 1))
                inter = sum(yp .* mask)
                denom = sum(yp) + sum(mask)
                sum_loss += 1 - (2 * inter + eps_t) / (denom + eps_t)
            end
        else
            error("dice_loss expects y_pred and y_true to have compatible shapes.")
        end
        return sum_loss / n_classes
    else
        if size(y_prob) != size(y_true_f)
            error("dice_loss expects y_pred and y_true to have the same shape.")
        end
        inter = sum(y_prob .* y_true_f)
        denom = sum(y_prob) + sum(y_true_f)
        return 1 - (2 * inter + eps_t) / (denom + eps_t)
    end
end
