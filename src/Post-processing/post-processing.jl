"""
`largest_connected_components(mask::Array{Int32, 3}, n_lcc::Int)`
"""
function largest_connected_components(mask::Array{Int32, 3}, n_lcc::Int)
	mask_gpu = CuArray(mask)
	labels_gpu = CUDA.fill(zero(Int32), size(mask))
	labels_next_gpu = CUDA.fill(zero(Int32), size(mask))

	dev = get_backend(labels_gpu)
	ndrange = size(mask)

	# 1. Initialize Labels
	# We pass only the arrays; the kernel infers dimensions
	event = initialize_labels_kernel(dev)(mask_gpu, labels_gpu; ndrange = ndrange)
	!isnothing(event) && wait(event)

	# 2. Propagate Labels
	# Use the largest dimension for worst-case propagation
	for _ in 1:maximum(ndrange)
		event = propagate_labels_kernel(dev)(mask_gpu, labels_gpu, labels_next_gpu; ndrange = ndrange)
		!isnothing(event) && wait(event)

		# Swap buffers for the next iteration
		labels_gpu, labels_next_gpu = labels_next_gpu, labels_gpu
	end

	labels_cpu = Array(labels_gpu)
	unique_labels = filter(x -> x != 0, unique(labels_cpu))

	if isempty(unique_labels)
		return []
	end

	# Calculate sizes and sort
	label_sizes = [(l, count(==(l), labels_cpu)) for l in unique_labels]
	sort!(label_sizes, by = x -> x[2], rev = true)

	top_n = min(n_lcc, length(label_sizes))
	return [labels_cpu .== label_sizes[i][1] for i in 1:top_n]
end

@kernel function initialize_labels_kernel(mask, labels)
	# Using Linear index for initialization is much faster
	idx = @index(Global, Linear)

	if mask[idx] == 1
		labels[idx] = Int32(idx)
	else
		labels[idx] = zero(Int32)
	end
end

@kernel function propagate_labels_kernel(mask, labels, labels_next)
	idx = @index(Global, Cartesian)
	# axes(mask) handles the bounds checks natively
	if idx in CartesianIndices(mask)
		if mask[idx] == 1
			min_label = labels[idx]

			# Check 26-connectivity (3x3x3 neighborhood)
			for i in -1:1, j in -1:1, k in -1:1
				(i == 0 && j == 0 && k == 0) && continue

				neighbor = idx + CartesianIndex(i, j, k)

				if neighbor in CartesianIndices(mask) && mask[neighbor] == 1
					if labels[neighbor] < min_label
						min_label = labels[neighbor]
					end
				end
			end
			labels_next[idx] = min_label
		else
			labels_next[idx] = zero(Int32)
		end
	end
end
