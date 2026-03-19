using ITKIOWrapper

"""
	fill_batch_tensor!(tensor5D, input_images, n_batches, n_channels)

Populate a 5-D tensor `(batch, channel, x, y, z)` with voxel data from a flat
`input_images` collection. Images are reshaped to `(n_batches, n_channels)` and
loaded into `tensor5D` in that order.

This is a clearer alias for the historic `batching_channeling!` function, which
remains available for backwards compatibility.
"""
function fill_batch_tensor!(tensor5D, input_images, n_batches, n_channels)
	expected_count = n_batches * n_channels
	actual_count = length(input_images)

	if actual_count != expected_count
		error("Mismatched input image count: expected $expected_count images ($n_batches batches × $n_channels channels), but got $actual_count")
	end
	
	for batch in 1:n_batches
		for channel in 1:n_channels
			# Calculate linear index: Row-Major Style
			# Batch 1, Channel 1 -> Index 1
			# Batch 1, Channel 2 -> Index 2
			idx = (batch - 1) * n_channels + channel

			image = input_images[idx]
			arr = image.voxel_data

			# Use (1,2,3) to keep spatial dimensions as (W, H, D)
			arr_perm = permutedims(arr, (1, 2, 3))

			# Corrected size check syntax
			if size(arr_perm) != size(tensor5D)[3:5]
				error("Size mismatch: got $(size(arr_perm)), expected $(size(tensor5D)[3:5])")
			end

			tensor5D[batch, channel, :, :, :] .= arr_perm
		end
	end
end

"""
	batching_channeling!(tensor5D, input_images, n_batches, n_channels)

Legacy name kept for backwards compatibility. Prefer `fill_batch_tensor!`.
"""
batching_channeling!(tensor5D, input_images, number_of_batches, number_of_channels) =
	fill_batch_tensor!(tensor5D, input_images, number_of_batches, number_of_channels)
