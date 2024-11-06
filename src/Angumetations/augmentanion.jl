include("./Utils.jl")
include("./Spatial_metadata_change.jl")
include("./Resample_to_target.jl")
using  Random, Statistics, CUDA, KernelAbstractions, Distributions

"""
`augment_brightness(image::Union{MedImage,Array{Float32, 3}}, value::Float64, mode::String)::Union{MedImage,Array{Float32, 3}}`

Adjust the brightness of an image either additively or multiplicatively.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `value`: The value to adjust the brightness by.
- `mode`: Adjustment mode, either "additive" which adds `value` to each voxel or "multiplicative" which multiplies each voxel by `value`.

# Returns
- `MedImage` or `Array`: The brightness-adjusted image, with the same type as the input.

# Errors
Throws an error if the mode is neither "additive" nor "multiplicative".
"""
function augment_brightness(image::Union{MedImage,Array{Float32, 3}}, value::Float64, mode::String)::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)

    # Store original min and max values before augmentation, excluding the minimum value
    original_min, original_max = extrema(im)
    masked_im = im[im .!= original_min]  # Mask out the minimum value from calculations
    masked_min, masked_max = extrema(masked_im)
    println("Masked min:", masked_min, "Masked max:", masked_max)

    # Apply augmentation, ignoring the minimum value
    if mode == "additive"
        im[im .!= original_min] .+= value  # Add value to each voxel except the minimum
    elseif mode == "multiplicative"
        im[im .!= original_min] .*= value  # Multiply each voxel except the minimum by value
    else
        error("Invalid mode. Choose 'additive' or 'multiplicative'.")
    end

    # Clamp to the original range (ignoring the minimum for clamping)
    im = clamp.(im, masked_min, masked_max)

    # Check min and max after clamping
    clamped_min, clamped_max = extrema(im)
    println("After clamping - Min:", clamped_min, "Max:", clamped_max)

    # Return the brightness-adjusted image with the same type as the input.
    return union_check_output(image, im)
end



"""
`augment_contrast(image::Union{MedImage,Array{Float32, 3}}, factor::Float64)::Union{MedImage,Array{Float32, 3}}`

Adjust the contrast of an image by scaling its pixel values around the mean.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `factor`: The scaling factor for contrast adjustment.

# Returns
- `MedImage` or `Array`: The contrast-adjusted image, with the same type as the input.
"""
function augment_contrast(image::Union{MedImage,Array{Float32, 3}}, factor::Float64)::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)
    original_min, original_max = extrema(im)    
    mn = mean(im)                       # Calculate the mean of the image.
    im .= (im .- mn) .* factor .+ mn    # Scale voxel values around the mean by factor.
    im = clamp.(im, original_min, original_max)
    # Return the contrast-adjusted image with the same type as the input.
    return union_check_output(image, im)
end

"""
`augment_gamma(image::Union{MedImage,Array{Float32, 3}}, gamma::Float64)::Union{MedImage,Array{Float32, 3}}`

Apply gamma correction to an image.
Gamma correction adjusts image luminance by applying a non-linear operation to voxel values.
It enhances shadow details with gamma < 1 and highlight details with gamma > 1, aligning image brightness with human visual perception.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `gamma`: The gamma correction factor.

# Returns
- `MedImage` or `Array`: Gamma-corrected image, with the same type as the input.
"""
function augment_gamma(image::Union{MedImage,Array{Float32, 3}}, gamma::Float64)::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)

    min_val, max_val = extrema(im)                              # Get the minimum and maximum voxel values.
    normalized_data = (im .- min_val) ./ (max_val - min_val)    # Normalize voxel values.
    transformed_data = normalized_data .^ gamma                 # Apply gamma correction.
    im .= (transformed_data .* (max_val - min_val)) .+ min_val  # Denormalize voxel values.

    # Return the gamma-corrected image with the same type as the input.
    return union_check_output(image, im)
end


"""
`augment_gaussian_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Union{MedImage,Array{Float32, 3}}`

Add Gaussian noise to an image.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `variance`: The variance of the Gaussian noise to add.

# Returns
- `Array{Float32, 3}`: Noise-augmented image, with the same type as the input.
"""
function augment_gaussian_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)
    original_min, original_max = extrema(im)
    noise = rand(Normal(0.0, variance), size(im))               # Generate Gaussian noise.
    im .+= noise                                                # Add noise to the image.                       
    im = clamp.(im, original_min, original_max)
    # Return the noisy image with the same type as the input.
    return union_check_output(image, im)
end

"""
`augment_rician_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Union{MedImage,Array{Float32, 3}}`

Add Rician noise to an image.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `variance`: The variance of the Rician noise.

# Returns
- `Array{Float32, 3}`: Noise-augmented image.
"""
function augment_rician_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)

    noise1 = rand(Normal(0.0, variance), size(im))            # Generate Gaussian noise.
    noise2 = rand(Normal(0.0, variance), size(im))            # Generate another Gaussian noise.
    im .= sqrt.((im .+ noise1).^2 + noise2.^2) .* sign.(im)   # Add Rician noise to the image.

    # Return the noisy image with the same type as the input.
    return union_check_output(image, im)
end


"""
`augment_mirror(image::Union{MedImage,Array{Float32, 3}}, axes=(1, 2, 3))::Union{MedImage,Array{Float32, 3}}`

Mirror an image along specified axes.
\nSupports `MedImage` or 3D `Float32` arrays.

# Arguments
- `image`: The input image.
- `axes`: A tuple indicating which axes to mirror (1, and/or 2, and/or 3).

# Returns
- `Array{Float32, 3}`: Mirrored image.
"""
function augment_mirror(image::Union{MedImage,Array{Float32, 3}}, axes=(1, 2, 3)::Tuple{Int, Int, Int})::Union{MedImage,Array{Float32, 3}}
    im = union_check_input(image)
    
    if 1 in axes
        im = im[end:-1:1, :, :]    # Mirror along the x-axis.
    end
    if 2 in axes
        im = im[:, end:-1:1, :]    # Mirror along the y-axis.
    end
    if 3 in axes
        im = im[:, :, end:-1:1]    # Mirror along the z-axis.
    end

    # Return the mirrored image with the same type as the input.
    return union_check_output(image, im)
end


# work in progress, this feature will not work in pipline
function augment_scaling(image::MedImage,scale_factor::Float64, interpolator_enum)::Array{Float32, 3} #change scale_factor to Tuple{Float64, Float64, Float64} and test it
    im = image.voxel_data
    original_size = size(im) 
    new_spacing = image.spacing .* (1/scale_factor)
    image_scaled = resample_to_spacing(image, new_spacing, interpolator_enum)
    new_size = size(image_scaled.voxel_data)

    if any(new_size .< original_size)
        pad_beg = Tuple((original_size .- new_size) .÷ 2)
        pad_end = Tuple(original_size .- new_size .- pad_beg)
        new_image = pad_mi(image_scaled, pad_beg, pad_end, extrapolate_corner_median(im))
    elseif any(new_size .> original_size)
        crop_beg = Tuple((new_size .- original_size) .÷ 2)
        crop_size = original_size
        new_image = crop_mi(image_scaled, crop_beg, crop_size, interpolator_enum)
    end
    
    return new_image
end

# work in progress
function elastic_deformation3d(image::Union{MedImage,Array{Float32, 3}}, strength::Float64, interpolator_enum)
    img = union_check_input(image)
    
    deformed_img = similar(img)

    # Inicjalizacja wektorów przesunięcia dla każdego punktu obrazu
    displacement_x = randn(size(img)...) * strength
    displacement_y = randn(size(img)...) * strength
    displacement_z = randn(size(img)...) * strength

    # Wybór interpolatora
    if interpolator_enum == :Nearest_neighbour_en
        itp = interpolate(img, BSpline(Constant()))
    elseif interpolator_enum == :Linear_en
        itp = interpolate(img, BSpline(Linear()))
    elseif interpolator_enum == :B_spline_en
        itp = interpolate(img, BSpline(Cubic(Line(OnGrid()))))
    else
        error("Nieznany typ interpolatora!")
    end

    # Wywołanie kernela
    kernel = elastic_deformation_kernel(CPU())(img, deformed_img, displacement_x, displacement_y, displacement_z, itp)

    return union_check_output(image, im)
end

@kernel function elastic_deformation_kernel(img, deformed_img, displacement_x, displacement_y, displacement_z, size_x, size_y, size_z, itp)
    x_global, y_global, z_global = @index(Global, Cartesian)  
    
    if 1 <= x_global <= size_x && 1 <= y_global <= size_y && 1 <= z_global <= size_z
        new_x = x_global + displacement_x[x_global, y_global, z_global]
        new_y = y_global + displacement_y[x_global, y_global, z_global]
        new_z = z_global + displacement_z[x_global, y_global, z_global]

        if 1 <= new_x <= size_x && 1 <= new_y <= size_y && 1 <= new_z <= size_z
            deformed_img[x_global, y_global, z_global] = itp([new_x, new_y, new_z])  
        else
            deformed_img[x_global, y_global, z_global] = 0  
        end
    end
end

"""
`augment_gaussian_blur(image::Union{MedImage,Array{Float32, 3}}, sigma::Float64, shape::String, kernel_size::Int, processing_unit::String)::Union{MedImage,Array{Float32, 3}}`

Apply a Gaussian blur to an image using a specified sigma, kernel size, and kernel shape.
\nSupports `MedImage` or 3D `Float32` arrays.
\nSupports processing on different units (CPU, GPU, AMD).

# Arguments
- `image`: The input image, either a `MedImage` or a 3D `Float32` array.
- `sigma`: The standard deviation of the Gaussian kernel.
- `shape`: The dimensionality of the kernel ("2D" or "3D").
- `kernel_size`: The size of the kernel.
- `processing_unit`: The processing unit to use ("CPU", "GPU", or "AMD").

# Returns
- `Union{MedImage,Array{Float32, 3}}`: The blurred image.

# Errors
- Throws an error if an invalid processing unit is specified.
"""
function augment_gaussian_blur(image, sigma, shape, kernel_size, processing_unit)
    im = union_check_input(image)                                  
    kernel = create_gaussian_kernel(sigma, kernel_size, shape)# Create the Gaussian kernel.

    if shape == "2D"
        pad_x, pad_y = size(kernel) .÷ 2
    elseif shape == "3D"
        pad_x, pad_y, pad_z = size(kernel) .÷ 2
    end

    stretch = (pad_x, pad_x, pad_x)
    kernel = backend_check(processing_unit, kernel)          # Check and convert kernel for the specified backend.
    padded_im = pad_mi_stretch(im, stretch)                  # Apply padding to the image with values from the edges.
    padded_im = backend_check(processing_unit, padded_im)    # Check and convert padded image for the specified backend.
    img_x, img_y, img_z = size(padded_im)
    ndrange = (img_x, img_y, img_z)                          # Define the range of the image for kernel function.
    result = similar(padded_im)                              # Create a storage for the result.
    result = backend_check(processing_unit, result)          # Check and convert result storage for the specified backend.
    

    # Check and convert the processing unit for the specified backend.
    if processing_unit == "GPU"
        dev = get_backend(padded_im)
    elseif processing_unit == "CPU"
        dev = CPU()
    elseif processing_unit == "AMD"
        dev = get_backend(padded_im)
    else    
        error("Invalid processing unit. Choose 'CPU', 'GPU', or 'AMD'.")
    end

    # Apply the convolution using the appropriate backend and shape of the kernel.
    if shape == "2D"
        kernel_event = padded_convolution_kernel(dev)(result, padded_im, kernel, pad_x, pad_y, ndrange = ndrange)
    elseif shape == "3D"
        kernel_event = padded_convolution_kernel_3D(dev)(result, padded_im, kernel, pad_x, pad_y, pad_z, ndrange = ndrange)
    end    
    result = Array(result)  # Convert the result to an array if needed.

    # Remove the padding from the result.
    if shape == "2D"
        final_result = result[1:end-(pad_x*2), 1:end-(pad_x*2), 1:end-(pad_x*2)]
    elseif shape == "3D"
        final_result = result[1:end-(pad_x*2), 1:end-(pad_x*2), 1:end-(pad_x*2)]
    end
    # Return the blurred image with the same type as the input.
    return union_check_output(image, im)
end

"""
`create_gaussian_kernel(sigma, kernel_size, shape="3D")`

Generate a Gaussian kernel with a given sigma and size, either in 2D or 3D.
Supporting function for `augment_gaussian_blur`.

# Arguments
- `sigma`: Standard deviation of the Gaussian kernel.
- `kernel_size`: Size of the kernel.
- `shape`: The dimensionality of the kernel ("2D" or "3D").

# Returns
- `Array`: The Gaussian kernel.
"""
function create_gaussian_kernel(sigma, kernel_size, shape="3D")
    if shape == "2D"    
        kernel_range = floor(Int, kernel_size / 2)  # Calculate the range of the kernel.
        kernel = [exp(-((x^2 + y^2) / (2 * sigma^2))) for x in -kernel_range:kernel_range, y in -kernel_range:kernel_range] # Create the kernel.
        kernel ./= sum(kernel) # Normalize the kernel.
        return kernel
    elseif shape == "3D"
        kernel_range = floor(Int, kernel_size / 2)  # Calculate the range of the kernel.
        kernel = [exp(-((x^2 + y^2 + z^2) / (2 * sigma^2))) for x in -kernel_range:kernel_range, y in -kernel_range:kernel_range, z in -kernel_range:kernel_range] # Create the kernel.
        kernel ./= sum(kernel) # Normalize the kernel.
        return kernel
    end
end

"""
Kernel function for `augment_gaussian_blur`.
"""
@kernel function padded_convolution_kernel(result, im, kernel, pad_x, pad_y)
    pad_z = pad_x
    idx = @index(Global, Cartesian)
    img_x, img_y, img_z = size(im)
    kernel_x, kernel_y = size(kernel)

    x, y, z = idx[1], idx[2], idx[3]

    ix_start = max(pad_x, x - pad_x)
    ix_end = min(img_x, x + pad_x)
    iy_start = max(pad_y, y - pad_y)
    iy_end = min(img_y, y + pad_y)
    iz_start = max(pad_z, z - pad_z)
    iz_end = min(img_z, z + pad_z)

    value = 0.0
    for ix = ix_start:ix_end
        for iy = iy_start:iy_end
            for iz = iz_start:iz_end
                m = ix - x + pad_x + 1
                n = iy - y + pad_y + 1
                p = iz - z + pad_z + 1 
                if m > 0 && m <= kernel_x && n > 0 && n <= kernel_y && p > 0
                    value += im[ix, iy, iz] * kernel[m, n]
                end
            end
        end
    end
    if x >= pad_x + 1 && x <= img_x - pad_x && y >= pad_y + 1 && y <= img_y - pad_y && z >= pad_z + 1 && z <= img_z - pad_z
        result[x - pad_x, y - pad_y, z - pad_z] = value
    end
end

"""
Kernel function for `augment_gaussian_blur`.
"""
@kernel function padded_convolution_kernel_3D(result, im, kernel, pad_x, pad_y, pad_z)
    idx = @index(Global, Cartesian)
    img_x, img_y, img_z = size(im)
    kernel_x, kernel_y, kernel_z = size(kernel)

    x, y, z = idx[1], idx[2], idx[3]

    # Rozszerzony zakres indeksów, aby uwzględnić padding
    ix_start = max(pad_x, x - pad_x)
    ix_end = min(img_x, x + pad_x)
    iy_start = max(pad_y, y - pad_y)
    iy_end = min(img_y, y + pad_y)
    iz_start = max(pad_z, z - pad_z)
    iz_end = min(img_z, z + pad_z)

    value = 0.0
    for ix = ix_start:ix_end
        for iy = iy_start:iy_end
            for iz = iz_start:iz_end
                m = ix - x + pad_x + 1
                n = iy - y + pad_y + 1
                p = iz - z + pad_z + 1 
                if m > 0 && m <= kernel_x && n > 0 && n <= kernel_y && p > 0 && p <= kernel_z
                    value += im[ix, iy, iz] * kernel[m, n, p]
                end
            end
        end
    end
    if x >= pad_x + 1 && x <= img_x - pad_x && y >= pad_y + 1 && y <= img_y - pad_y && z >= pad_z + 1 && z <= img_z - pad_z
        result[x - pad_x, y - pad_y, z - pad_z] = value
    end
end

# work in progress, this feature do not work in pipline
function augment_simulate_low_resolution(image::Union{MedImage,Array{Float32, 3}}, blur_sigma::Float64, kernel_size::Int, downsample_scale::Float64)
    #im = union_check(image)
    blurred_voxel_data = augment_gaussian_blur(image, blur_sigma, kernel_size)
    image_downsampled = augment_scaling(blurred_voxel_data, downsample_scale)
    image_upsampled = augment_scaling(image_downsampled, 1/downsample_scale)

    return image_upsampled
end