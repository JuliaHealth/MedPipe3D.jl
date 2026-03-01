using KernelAbstractions

@kernel function erode_kernel!(output, input, struct_element)
    I = @index(Global, Cartesian)
    b, c, k, i, j = Tuple(I)

    offset_d = div(size(struct_element)[1], 2)
    offset_h = div(size(struct_element)[2], 2)
    offset_w = div(size(struct_element)[3], 2)

    if b <= size(input)[1] && c <= size(input)[2] &&  k <= size(input)[3] &&
        i <= size(input)[4] && j <= size(input)[5]

        result = true

        for m in 1:size(struct_element)[1]
            for n in 1:size(struct_element)[2]
                for p in 1:size(struct_element)[3]
                    nk = k + m - offset_d - 1
                    ni = i + n - offset_h - 1
                    nj = j + p - offset_w 

                    if 1 <= nk <= size(input)[3] && 1 <= ni <= size(input)[4] && 
                        nj <= size(input)[5]
                        if struct_element[m, n, p] == 1 && input[b, c, nk, ni, nj] == 0
                            result = false
                            break
                        end
                    else
                        # Treat out-of-bounds as 0 (zero-padding erosion)
                        if struct_element[m, n, p] == 1
                            result = false
                            break
                        end
                    end
                end
                if !result; break; end
            end
            if !result; break; end
        end
        @inbounds output[b, c, k, i, j] = result ? 1 : 0
    end
end

function erode!(output, input, struct_element)
    backend = get_backend(input)
    kernel! = erode_kernel!(backend)
    kernel!(output, input, struct_element, ndrange=size(output))
    return output
end