using Test
using HDF5
import MedImages

try
    using MedPipe3D
catch
    include(joinpath(@__DIR__, "..", "src", "MedPipe3D.jl"))
    using .MedPipe3D
end

@testset "Dataset → HDF5 conversion" begin

repo_root = joinpath(@__DIR__, "..")

data_dir   = joinpath(repo_root, "dataset", "raw", "Task02_Heart")
images_dir = joinpath(data_dir, "imagesTr")
labels_dir = joinpath(data_dir, "labelsTr")

hdf5_dir   = joinpath(repo_root, "dataset", "HDF5")
mkpath(hdf5_dir)

pathToHDF5 = joinpath(hdf5_dir, "heart_dataset.hdf5")

@test isdir(images_dir)
@test isdir(labels_dir)

function strip_nii_ext(path)
    base = basename(path)
    if endswith(base,".nii.gz")
        return replace(base,".nii.gz"=>"")
    elseif endswith(base,".nii")
        return replace(base,".nii"=>"")
    end
    return splitext(base)[1]
end

image_files = filter(f -> !startswith(basename(f), ".") && !startswith(basename(f), "._") && (endswith(f,".nii")||endswith(f,".nii.gz")),
                     readdir(images_dir; join=true))

label_files = filter(f -> !startswith(basename(f), ".") && !startswith(basename(f), "._") && (endswith(f,".nii")||endswith(f,".nii.gz")),
                     readdir(labels_dir; join=true))

@test length(image_files) > 0
@test length(label_files) > 0

image_map = Dict(strip_nii_ext(f)=>f for f in image_files)
label_map = Dict(strip_nii_ext(f)=>f for f in label_files)

common = intersect(keys(image_map),keys(label_map))

@test length(common) > 0

h5open(pathToHDF5,"w") do fid

    for (i,k) in enumerate(common)

        # Safely load the image depending on MedImages version
        function safe_load_image(path)
            if isdefined(MedImages, :load_images)
                return MedImages.load_images(path)[1]
            elseif isdefined(MedImages, :load_image) && hasmethod(MedImages.load_image, Tuple{String, String})
                return MedImages.load_image(path, "CT")
            elseif isdefined(MedImages, :load_image)
                return MedImages.load_image(path)
            end
            error("No load_image available")
        end

        img = safe_load_image(image_map[k])
        lbl = safe_load_image(label_map[k])

        img_data = reshape(Float32.(img.voxel_data), size(img.voxel_data)...,1)
        lbl_data = reshape(UInt32.(lbl.voxel_data .== 1), size(lbl.voxel_data)...,1)

        grp = create_group(fid,string(i-1))
        imgs = create_group(grp,"images")
        msks = create_group(grp,"masks")

        imgs["data"] = img_data
        msks["data"] = lbl_data

        # Add metadata required by convert_hdf5_to_medimages
        img_meta_group = create_group(imgs, "metadata")
        mask_meta_group = create_group(msks, "metadata")
        img_meta_0 = create_group(img_meta_group, "0")
        mask_meta_0 = create_group(mask_meta_group, "0")

        img_meta = Dict("file_path" => image_map[k], "image_type" => "CT", "spacing" => img.spacing, "origin" => img.origin, "direction" => img.direction)
        mask_meta = Dict("file_path" => label_map[k], "image_type" => "CT", "spacing" => lbl.spacing, "origin" => lbl.origin, "direction" => lbl.direction)

        MedPipe3D.safe_write_meta(img_meta_0, img_meta)
        MedPipe3D.safe_write_meta(mask_meta_0, mask_meta)
    end
end

@test isfile(pathToHDF5)

h5open(pathToHDF5,"r") do f
    @test length(keys(f)) > 0
end

end