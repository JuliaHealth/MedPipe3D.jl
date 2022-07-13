module LoadFromMonai

using PythonCall

function getMonaiObject()
    return pyimport("monai")
end    


function getSimpleItkObject()
    return pyimport("sitk")
end  

function permuteAndReverseFromMonai(pixels)
    sizz=size(pixels)
    for i in 1:sizz[2]
        for j in 1:sizz[3]
            pixels[:,i,j] =  reverse(pixels[:,i,j])
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
    imagePath
    ,labelPath
    ,trAnsforms=[])
    monai=pyimport("monai")
    #default transforms
    if(length(trAnsforms)==0)
        trAnsforms= [
        #monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Spacingd(keys=["image", "label"], pixdim=(  1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"])
        ]
    end
    trAnsformsComposed= monai.transforms.Compose(trAnsforms)

    dicttt= pydict(Dict([("image", imagePath),( "label",  labelPath )]))
    loadObj=monai.transforms.LoadImaged(keys=["image", "label"],reader= "ITKReader")(dicttt)
    metaData= pyconvert(Dict,pyconvert(Dict,loadObj)["image_meta_dict"])

    loadObj = trAnsformsComposed(loadObj)

    image = permuteAndReverseFromMonai(pyconvert(Array,loadObj["image"].detach().numpy()[0]))
    label =permuteAndReverseFromMonai(pyconvert(Array,loadObj["label"].detach().numpy()[0]))
    
return (image,label,metaData)
    
end



"""
resample to given size using sitk
"""

function resamplesitkImageTosize(image,targetSpac)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetSize(new_size)
    return resample.Execute(image)

end



"""
given file paths it loads 
imagePath - path to main image
labelPath - path to label

"""
function loadBySitkromImageAndLabelPaths(
    imagePath
    ,labelPath
    ,targetSpacing)

    sitk=getSimpleItkObject()
    
    image=sitk.ReadImage(imagePath)
    label=sitk.ReadImage(labelPath)

    image=sitk.DICOMOrient(image, "RAS")
    label=sitk.DICOMOrient(label, "RAS")

    image=resamplesitkImageTosize(image,targetSpacing)
    label=resamplesitkImageTosize(label,targetSpacing)

    imageArr=sitk.GetArrayFromImage(image)
    labelArr=sitk.GetArrayFromImage(label)

    imageSize=image.GetSize()
    labelSize= label.GetSize()

return (imageArr,labelArr,metaData,imageSize,labelSize)
    
end


end
