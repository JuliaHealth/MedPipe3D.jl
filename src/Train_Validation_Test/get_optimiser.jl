function get_optimiser(name::String)
    name = lowercase(name)
    if name == "descent"
        return Optimisers.Descent()
    elseif name == "momentum"
        return Optimisers.Momentum()
    elseif name == "nesterov"
        return Optimisers.Nesterov()
    elseif name == "rprop"
        return Optimisers.Rprop()
    elseif name == "rmsprop"
        return Optimisers.RMSProp()
    elseif name == "adam"
        return Optimisers.Adam()
    elseif name == "radam"
        return Optimisers.RAdam()
    elseif name == "adamax"
        return Optimisers.AdaMax()
    elseif name == "oadam"
        return Optimisers.OAdam()
    elseif name == "adagrad"
        return Optimisers.AdaGrad()
    elseif name == "adadelta"
        return Optimisers.AdaDelta()
    elseif name == "amsgrad"
        return Optimisers.AMSGrad()
    elseif name == "nadam"
        return Optimisers.NAdam()
    elseif name == "adamw"
        return Optimisers.AdamW()
    elseif name == "adabelief"
        return Optimisers.AdaBelief()
    elseif name == "lion"
        return Optimisers.Lion()
    else
        error("Unknown optimiser: $name")
    end
end