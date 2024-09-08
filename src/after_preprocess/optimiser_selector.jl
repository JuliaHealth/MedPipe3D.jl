export get_optimiser
using Optimisers

"""
Selects an optimisation algorithm based on the supplied string.

# Arguments
- `name::String`: The name of the optimisation algorithm.
- `config_optimiser_args`: Arguments to initialize the selected optimisation algorithm.

# Returns
- An instance of the selected optimisation algorithm.

# Example
```julia
optimiser = get_optimiser("Adam", config_optimiser_args)
"""
function get_optimiser(name::String, config_optimiser_args)
    name = lowercase(name)
    if name == "descent"
        return Optimisers.Descent(config_optimiser_args...)
    elseif name == "momentum"
        return Optimisers.Momentum(config_optimiser_args...)
    elseif name == "nesterov"
        return Optimisers.Nesterov(config_optimiser_args...)
    elseif name == "rprop"
        return Optimisers.Rprop(config_optimiser_args...)
    elseif name == "rmsprop"
        return Optimisers.RMSProp(config_optimiser_args...)
    elseif name == "adam"
        return Optimisers.Adam(config_optimiser_args...)
    elseif name == "radam"
        return Optimisers.RAdam(config_optimiser_args...)
    elseif name == "adamax"
        return Optimisers.AdaMax(config_optimiser_args...)
    elseif name == "oadam"
        return Optimisers.OAdam(config_optimiser_args...)
    elseif name == "adagrad"
        return Optimisers.AdaGrad(config_optimiser_args...)
    elseif name == "adadelta"
        return Optimisers.AdaDelta(config_optimiser_args...)
    elseif name == "amsgrad"
        return Optimisers.AMSGrad(config_optimiser_args...)
    elseif name == "nadam"
        return Optimisers.NAdam(config_optimiser_args...)
    elseif name == "adamw"
        return Optimisers.AdamW(config_optimiser_args...)
    elseif name == "adabelief"
        return Optimisers.AdaBelief(config_optimiser_args...)
    elseif name == "lion"
        return Optimisers.Lion(config_optimiser_args...)
    else
        error("Unknown optimiser: $name")
    end
end