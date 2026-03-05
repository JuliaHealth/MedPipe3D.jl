"""
`union_check_input(image)::Array{Float32, 3}`

Extract voxel data from MedImage or return array directly.
"""
function union_check_input(image)::Array{Float32, 3}
    # Check if MedImage type exists and image is an instance of it
    if @isdefined(MedImage) && image isa MedImage
        return image.voxel_data
    else
        return image
    end
end

"""
`get_spacing(image)::Tuple`

Get spacing from MedImage or return default spacing for Array.
"""
function get_spacing(image)::Tuple
    if @isdefined(MedImage) && image isa MedImage
        return image.spacing
    else
        return (1.0, 1.0, 1.0)
    end
end

"""
`union_check_output(original, processed::Array{Float32, 3})`

Convert processed array back to original type.
"""
function union_check_output(original, processed::Array{Float32, 3})
    # Check if MedImage type exists and original is an instance of it
    if @isdefined(MedImage) && original isa MedImage
        return MedImage(processed, original.spacing, original.origin, original.direction)
    else
        return processed
    end
end

"""
`original_min(image::Array{Float32, 3})::Float32`

Get the minimum value in the image (excluding NaN/Inf).
"""
function original_min(image::Array{Float32, 3})::Float32
    return minimum(filter(x -> !isnan(x) && !isinf(x), image))
end

"""
`pad_mi(image, pad_beg::Tuple, pad_end::Tuple, value::Float32)`

Pad an image with specified value.
"""
function pad_mi(image, pad_beg::Tuple, pad_end::Tuple, value::Float32)
    im = union_check_input(image)
    sz = size(im)
    new_sz = sz .+ pad_beg .+ pad_end
    padded = fill(value, Float32.(new_sz) .|> Int)
    padded[pad_beg[1]+1:pad_beg[1]+sz[1],
           pad_beg[2]+1:pad_beg[2]+sz[2],
           pad_beg[3]+1:pad_beg[3]+sz[3]] .= im
    return union_check_output(image, Float32.(padded))
end

"""
`pad_mi_stretch(image::Array{Float32, 3}, stretch::Tuple)::Array{Float32, 3}`

Pad image by replicating edge values for convolution.
"""
function pad_mi_stretch(image::Array{Float32, 3}, stretch::Tuple)::Array{Float32, 3}
    sx, sy, sz = size(image)
    px, py, pz = stretch
    new_sx, new_sy, new_sz = sx + 2*px, sy + 2*py, sz + 2*pz
    padded = zeros(Float32, new_sx, new_sy, new_sz)
    # Copy original data into center
    padded[px+1:px+sx, py+1:py+sy, pz+1:pz+sz] .= image
    # Replicate edges
    for i in 1:px
        padded[i, py+1:py+sy, pz+1:pz+sz] .= image[1, :, :]
        padded[px+sx+i, py+1:py+sy, pz+1:pz+sz] .= image[end, :, :]
    end
    for j in 1:py
        padded[:, j, pz+1:pz+sz] .= padded[:, py+1, pz+1:pz+sz]
        padded[:, py+sy+j, pz+1:pz+sz] .= padded[:, py+sy, pz+1:pz+sz]
    end
    for k in 1:pz
        padded[:, :, k] .= padded[:, :, pz+1]
        padded[:, :, pz+sz+k] .= padded[:, :, pz+sz]
    end
    return padded
end


"""
`crop_mi(image, crop_beg::Tuple, crop_size::Tuple)`

Crop an image.
"""
function crop_mi(image, crop_beg::Tuple, crop_size::Tuple)
    im = union_check_input(image)
    cropped = im[crop_beg[1]:crop_beg[1]+crop_size[1]-1,
                 crop_beg[2]:crop_beg[2]+crop_size[2]-1,
                 crop_beg[3]:crop_beg[3]+crop_size[3]-1]
    return union_check_output(image, cropped)
end

"""
`crop_mi(image, crop_beg::Tuple, crop_size::Tuple, interpolator_enum)`

Crop an image with interpolator.
"""
function crop_mi(image, crop_beg::Tuple, crop_size::Tuple, interpolator_enum)
    im = union_check_input(image)
    cropped = im[crop_beg[1]:crop_beg[1]+crop_size[1]-1,
                 crop_beg[2]:crop_beg[2]+crop_size[2]-1,
                 crop_beg[3]:crop_beg[3]+crop_size[3]-1]
    return union_check_output(image, cropped)
end

"""
`extrapolate_corner_median(image::Array{Float32, 3})::Float32`

Get median value for corner extrapolation.
"""
function extrapolate_corner_median(image::Array{Float32, 3})::Float32
    return median(filter(x -> !isnan(x) && !isinf(x), image))
end

"""
`backend_check(processing_unit::String, data)`

Check and convert data for specified backend.
"""
function backend_check(processing_unit::String, data)
    if processing_unit == "GPU"
        return CuArray(data)
    elseif processing_unit == "AMD"
        # AMD GPU support would go here
        return data
    else
        return data
    end
end

"""
`get_backend(data)`

Get backend from data.
"""
function get_backend(data)
    if data isa CuArray
        return CUDABackend()
    else
        return CPU()
    end
end