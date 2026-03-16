using HDF5

# ────────────────────────────────────────────────────────────
# HDF5 structure printer
# ────────────────────────────────────────────────────────────

"""
    print_hdf5_contents(hdf5_path)

Pretty-print the full group / dataset / attribute tree of the HDF5 file at
`hdf5_path`. Useful for quick inspection of pipeline output files.

Output format (indented):
```
- Group: /
    Attribute: key = value
    - Dataset: data
        Shape: (128, 128, 64)
        Attribute: spacing = [1.0, 1.0, 2.0]
```
"""
function print_hdf5_contents(hdf5_path::String)
    function _print_node(name::String, obj, indent::Int)
        pad = "  "^indent
        if isa(obj, Union{HDF5.Group, HDF5.File})
            println("$pad- Group: $name")
            for attr_name in keys(attrs(obj))
                println("$pad    Attribute: $attr_name = $(read_attribute(obj, attr_name))")
            end
            for member_name in keys(obj)
                _print_node(member_name, obj[member_name], indent + 1)
            end
        elseif isa(obj, HDF5.Dataset)
            println("$pad- Dataset: $name")
            println("$pad    Shape: $(size(obj))")
            for attr_name in keys(attrs(obj))
                println("$pad    Attribute: $attr_name = $(read_attribute(obj, attr_name))")
            end
        else
            println("$pad- Unknown: $name")
        end
    end

    h5open(hdf5_path, "r") do file
        println("HDF5 contents of '$hdf5_path':")
        _print_node("/", file, 0)
    end
end