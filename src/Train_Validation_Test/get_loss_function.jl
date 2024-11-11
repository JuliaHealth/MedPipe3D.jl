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
    else
        error("Unsupported or unrecognized loss function: $name. You may need to define this loss function manually.")
    end
end