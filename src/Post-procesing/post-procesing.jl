
@kernel function initialize_labels_kernel(mask, labels, width, height, depth)
    idx = @index(Global, Cartesian)
    i = idx[1]
    j = idx[2]
    k = idx[3]
    
    if i >= 1 && i <= width && j >= 1 && j <= height && k >= 1 && k <= depth
        if mask[i, j, k] == 1
            labels[i, j, k] = i + (j - 1) * width + (k - 1) * width * height
        else
            labels[i, j, k] = 0
        end
    end
end

@kernel function propagate_labels_kernel(mask, labels, width, height, depth)
    idx= @index(Global, Cartesian)
    i = idx[1]
    j = idx[2]
    k = idx[3]

    if i >= 1 && i <= width && j >= 1 && j <= height && k >= 1 && k <= depth
        if mask[i, j, k] == 1
            current_label = labels[i, j, k]
            for di in -1:1
                for dj in -1:1
                    for dk in -1:1
                        if di == 0 && dj == 0 && dk == 0
                            continue
                        end
                        ni = i + di
                        nj = j + dj
                        nk = k + dk
                        if ni >= 1 && ni <= width && nj >= 1 && nj <= height && nk >= 1 && nk <= depth
                            if mask[ni, nj, nk] == 1 && labels[ni, nj, nk] < current_label
                                labels[i, j, k] = labels[ni, nj, nk]
                            end
                        end
                    end
                end
            end
        end
    end
end

function largest_connected_components(mask::Array{Int32, 3}, n_lcc::Int)
    width, height, depth = size(mask)
    mask_gpu = CuArray(mask)
    labels_gpu = CUDA.fill(0, size(mask))
    dev = get_backend(labels_gpu)
    ndrange = (width, height, depth)
    workgroupsize = (3, 3, 3)

    # Initialize labels
    initialize_labels_kernel(dev)(mask_gpu, labels_gpu, width, height, depth, ndrange = ndrange)
    CUDA.synchronize()

    # Propagate labels iteratively
    for _ in 1:10 
        propagate_labels_kernel(dev, workgroupsize)(mask_gpu, labels_gpu, width, height, depth, ndrange = ndrange)
        CUDA.synchronize()
    end

    # Download labels back to CPU
    labels_cpu = Array(labels_gpu)
    
    # Find all unique labels and their sizes
    unique_labels = unique(labels_cpu)
    label_sizes = [(label, count(labels_cpu .== label)) for label in unique_labels if label != 0]

    # Sort labels by size and get the top n_lcc
    sort!(label_sizes, by = x -> x[2], rev = true)
    top_labels = label_sizes[1:min(n_lcc, length(label_sizes))]

    # Create a mask for each of the top n_lcc components
    components = [labels_cpu .== label[1] for label in top_labels]
    return components
end