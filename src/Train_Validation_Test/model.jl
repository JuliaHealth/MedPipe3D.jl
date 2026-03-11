"""
`create_segmentation_model(num_classes, in_channels)`

Creates a lightweight 3D segmentation model compatible with the HDF5 data layout
(X, Y, Z, C, N). The model outputs `num_classes` channels.
"""
function create_segmentation_model(num_classes::Int, in_channels::Int)
    return Lux.Chain(
        Lux.Conv((3, 3, 3), in_channels => 8, pad=1),
        Lux.InstanceNorm(8),
        Lux.relu,
        Lux.Conv((3, 3, 3), 8 => 16, pad=1),
        Lux.InstanceNorm(16),
        Lux.relu,
        Lux.Conv((1, 1, 1), 16 => num_classes, pad=0)
    )
end
