"""
Training utilities for Lux 0.5/1.x.

Defines:
- `single_train_step!` compatible with current code.
"""

function single_train_step!(::ADTypes.AutoZygote, loss_function, data, tstate)
    x, y = data

    # Move data to the same device as the parameters/states
    dev = MLDataDevices.get_device(tstate.parameters)
    x = x |> dev
    y = y |> dev

    ps = tstate.parameters
    st = tstate.states

    # Ensure parameters are on the same device as data (in case they were moved externally)
    ps = ps |> dev
    st = st |> dev

    obj_fn = function (model, ps_, st_, data_)
        x_, y_ = data_
        y_pred, st_new = Lux.apply(model, x_, ps_, st_)
        loss = loss_function(y_pred, y_)
        return loss, st_new, (; y_pred = y_pred)
    end

    _, loss, stats, new_tstate = Lux.Training.single_train_step!(ADTypes.AutoZygote(), obj_fn, (x, y), tstate)
    y_pred = get(stats, :y_pred, nothing)
    return y_pred, loss, stats, new_tstate
end

function single_train_step!(ad::ADTypes.AbstractADType, loss_function, data, tstate)
    if ad isa ADTypes.AutoZygote
        return single_train_step!(ADTypes.AutoZygote(), loss_function, data, tstate)
    end
    error("Unsupported AD backend: $(typeof(ad)). Only AutoZygote is supported in this pipeline.")
end
