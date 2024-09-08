
"""
    log_metric(logger::TBLogger, name::AbstractString, value::Number, epoch::Int)

Logs a scalar metric `value` under the tag `name` for the given `epoch`.
"""
function log_metric(logger::TBLogger, name::AbstractString, value::Number, epoch::Int)
    log_value(logger, name, value, step=epoch)
end


"""
    log_vector(logger::TBLogger, name::AbstractString, vector::AbstractVector{<:Number}, step::Int)

Logs a vector of numbers `vector` under the tag [`name`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fmedia%2Fjm%2FhddData%2Fprojects%2FTensorBoardLogger.jl%2Fsrc%2FLoggers%2FLogImage.jl%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A25%2C%22character%22%3A38%7D%5D "src/Loggers/LogImage.jl") for the given [`step`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fmedia%2Fjm%2FhddData%2Fprojects%2FTensorBoardLogger.jl%2Fsrc%2FLoggers%2FLogImage.jl%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A25%2C%22character%22%3A86%7D%5D "src/Loggers/LogImage.jl").
"""
function log_vector(logger::TBLogger, name::AbstractString, vector::AbstractVector{<:Number}, step::Int)
    log_histogram(logger, name, vector, step=step)
end


"""
    log_image(logger::TBLogger, name::AbstractString, image::AbstractArray{<:Colorant}, step::Int)

Logs a 2-dimensional image `image` under the tag `name` for the given `step`.
"""
function log_image(logger::TBLogger, name::AbstractString, image::AbstractArray{<:Colorant}, step::Int)
    log_image(logger, name, image, step=step)
end



"""
Saves the given results to a file. using Medimages.jl, gets approriate metadata from metadata_ref

# Arguments
- `results`: The results data to be saved.
- `filename`: The name of the file to which the results are to be saved.
"""
function save_results(results, filename, metadata_ref)
    Medimages.save(results, filename, format)


end

