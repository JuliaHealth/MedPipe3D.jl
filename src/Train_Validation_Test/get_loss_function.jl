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