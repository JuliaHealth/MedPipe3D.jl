using KernelAbstractions
"""
# Description:
The kernel performs binary dilation on a 5D tensor of shape
[batch_size, channels, depth, height, width] using a given
structuring element. Each voxel is checked against its neighborhood
as defined by the structuring element.

"""
@kernel function dilate_kernel(output, input, struct_element)
    I = @index(Global, Cartesian)
    b, c, k, i, j = Tuple(I)
    offset_h = div(size(struct_element)[1], 2)
    offset_w = div(size(struct_element)[2], 2)
    offset_d = div(size(struct_element)[3], 2)

    if b <= size(input)[1] && c <= size(input)[2] && k <= size(input)[3] &&
        i <= size(input)[4] && j <= size(input)[5]

        result = false

        for m in 1:size(struct_element)[1]
            for n in 1:size(struct_element)[2]
                for p in 1:size(struct_element)[3]
                    # actual coordinates in the 3D input image within the current batch and channel slice
                    # that the structuring element is currently influencing
                    ni = i + m - offset_h - 1 
                    nj = j + n - offset_w - 1
                    nk = k + p - offset_d - 1

                    if 1 <= nk <= size(input)[3] && 1 <= ni <= size(input)[4] &&
                        1 <= nj <= size(input)[5]

                        if input[b, c, nk, ni, nj] == 1 && struct_element[m, n, p] == 1
                            result = true
                            break
                        end
                    end
                end
                if result; break; end
            end
            if result; break; end
        end
        
        @inbounds output[b, c, k, i, j] = result ? 1 : 0
    end
end

function dilate!(output, input, struct_element)

    backend = get_backend(input)
    kernel! = dilate_kernel!(backend)
    kernel!(output, input, struct_element, ndrange=size(output))

    return output
end
