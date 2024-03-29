module LoadFromMonai

using PythonCall
# using MedEye3d.visualizationFromHdf5

function getMonaiObject()
    return pyimport("monai")
end


function getSimpleItkObject()
    return pyimport("SimpleITK")
end

function myPyconvert(typeA, obj)
    return pyconvert(typeA, obj)
end

function permuteAndReverseFromMonai(pixels)
    sizz = size(pixels)
    for i in 1:sizz[2]
        for j in 1:sizz[3]
            pixels[:, i, j] = reverse(pixels[:, i, j])
        end# 
    end# 
    return pixels
end#permuteAndReverse


"""
given file paths it loads 
imagePath - path to main image
labelPath - path to label
transforms - monai.transforms.Compose object -important Load imaged should not be in this list it is added separately
default transforms standardize orientation, voxel dimensions crops unnecessary background reducing array size
ensure type of the images and labels so it will be easily convertible to for example numpy and then Julia
more in https://docs.monai.io/en/stable/transforms.html
"""
function loadByMonaiFromImageAndLabelPaths(
    imagePath, labelPath, trAnsforms=[])
    monai = pyimport("monai")
    #default transforms
    if (length(trAnsforms) == 0)
        trAnsforms = [
            #monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"])
        ]
    end
    trAnsformsComposed = monai.transforms.Compose(trAnsforms)

    dicttt = pydict(Dict([("image", imagePath), ("label", labelPath)]))
    loadObj = monai.transforms.LoadImaged(keys=["image", "label"], reader="ITKReader")(dicttt)
    metaData = pyconvert(Dict, pyconvert(Dict, loadObj)["image_meta_dict"])

    loadObj = trAnsformsComposed(loadObj)

    image = permuteAndReverseFromMonai(pyconvert(Array, loadObj["image"].detach().numpy()[0]))
    label = permuteAndReverseFromMonai(pyconvert(Array, loadObj["label"].detach().numpy()[0]))

    return (image, label, metaData)

end



"""
resample to given size using sitk
"""

function resamplesitkImageTosize(image, targetSpac, sitk, interpolator)

    orig_spacing = pyconvert(Array, image.GetSpacing())
    origSize = pyconvert(Array, image.GetSize())

    new_size = (Int(round(origSize[1] * (orig_spacing[1] / targetSpac[1]))),
        Int(round(origSize[2] * (orig_spacing[2] / targetSpac[2]))),
        Int(round(origSize[3] * (orig_spacing[3] / targetSpac[3]))))

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    resample.SetSize(new_size)
    return resample.Execute(image)

end



function permuteAndReverseFromSitk(pixels)
    pixels = permutedims(pixels, (3, 2, 1))
    sizz = size(pixels)
    for i in 1:sizz[2]
        for j in 1:sizz[3]
            pixels[:, i, j] = reverse(pixels[:, i, j])
        end# 
    end# 
    return pixels
end#permuteAndReverse


"""
given file paths it loads 
imagePath - path to main image
labelPath - path to label
also it make the spacing equal to target spacing and the orientation as RAS
"""
function loadBySitkromImageAndLabelPaths(
    imagePath, labelPath, targetSpacing=(1, 1, 1))

    sitk = getSimpleItkObject()

    image = sitk.ReadImage(imagePath)
    label = sitk.ReadImage(labelPath)

    image = sitk.DICOMOrient(image, "RAS")
    label = sitk.DICOMOrient(label, "RAS")

    image = resamplesitkImageTosize(image, targetSpacing, sitk, sitk.sitkBSpline)
    label = resamplesitkImageTosize(label, targetSpacing, sitk, sitk.sitkNearestNeighbor)

    imageArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(image)))
    labelArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(label)))

    imageSize = image.GetSize()
    labelSize = label.GetSize()


    return (imageArr, labelArr, imageSize, imageSize, labelSize)

end

"""
padd with given value symmetrically to get the predifined target size and return padded image
"""
function padToSize(image1, targetSize, paddValue, sitk)
    currentSize = pyconvert(Array, image1.GetSize())
    sizediffs = (targetSize[1] - currentSize[1], targetSize[2] - currentSize[2], targetSize[3] - currentSize[3])
    halfDiffSize = (Int(floor(sizediffs[1] / 2)), Int(floor(sizediffs[2] / 2)), Int(floor(sizediffs[3] / 2)))
    rest = (sizediffs[1] - halfDiffSize[1], sizediffs[2] - halfDiffSize[2], sizediffs[3] - halfDiffSize[3])
    #print(f" currentSize {currentSize} targetSize {targetSize} halfDiffSize {halfDiffSize}  rest {rest} paddValue {paddValue} sizediffs {type(sizediffs)}")

    # halfDiffSize=()
    # rest=zeros(Int,rest)

    return sitk.ConstantPad(image1, halfDiffSize, rest, paddValue)
    #return sitk.ConstantPad(image1, (1,1,1), (1,1,1), paddValue)
end #padToSize


"""
given file paths it loads 
imagePath - path to main image
labelPath - path to label
also it make the spacing equal to target spacing and the orientation as RAS
in the end pad to target size
"""
function loadandPad(
    imagePath, labelPath, targetSpacing, targetSize)

    sitk = getSimpleItkObject()

    image = sitk.ReadImage(imagePath)
    label = sitk.ReadImage(labelPath)

    image = sitk.DICOMOrient(image, "RAS")
    label = sitk.DICOMOrient(label, "RAS")

    image = resamplesitkImageTosize(image, targetSpacing, sitk, sitk.sitkBSpline)
    label = resamplesitkImageTosize(label, targetSpacing, sitk, sitk.sitkNearestNeighbor)

    image = padToSize(image, targetSize, 0, sitk)
    label = padToSize(label, targetSize, 0, sitk)

    imageArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(image)))
    labelArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(label)))

    imageSize = pyconvert(Array, image.GetSize())
    labelSize = pyconvert(Array, label.GetSize())


    return (imageArr, labelArr, imageSize, labelSize)

end



"""
given file path it loads 
imagePath - path to main image
also it make the spacing equal to target spacing and the orientation as RAS
in the end pad to target size
if any target size entry is -1 one will keep the original size in this dimension
"""
function loadandPadSingle(
    imagePath, targetSpacing, targetSize, isLabel)

    sitk = getSimpleItkObject()

    image = sitk.ReadImage(imagePath)

    image = sitk.DICOMOrient(image, "RAS")
    interpolator = sitk.sitkBSpline
    if (isLabel)
        interpolator = sitk.sitkNearestNeighbor
    end

    image = resamplesitkImageTosize(image, targetSpacing, sitk, interpolator)
    imageSize = pyconvert(Array, image.GetSize())
    targetSize = [i for i in targetSize]
    # in case some size is set to -1 it marks just that it should not be changed
    for i in 1:3
        if (targetSize[i] < 0)
            targetSize[i] = imageSize[i]
        end
    end#for
    image = padToSize(image, targetSize, 0, sitk)

    imageArr = permuteAndReverseFromSitk(pyconvert(Array, sitk.GetArrayFromImage(image)))

    imageSize = pyconvert(Array, image.GetSize())


    return (imageArr, imageSize)

end




end

