module HDF5saveUtils
using HDF5

export saveManualModif,saveMaskbyName





"""
saving to the HDF5 data from manual modifications
requires the manual modification array to be called manualModif
fid- object referencing HDF5 database
"""
function saveManualModif(fid,patienGroupName , mainScrollDat)
    manualModif= filter((it)->it.name=="manualModif" ,mainScrollDat.dataToScroll)[1].dat
    group = fid[patienGroupName]
    if(!haskey(group, "manualModif"))
        write(group, "manualModif", manualModif)
        dset = group["manualModif"]
        write_attribute(dset, "dataType", "manualModif")
        write_attribute(dset, "min", minimum(manualModif))
        write_attribute(dset, "max", max(maximum(manualModif), 1))

    else
        delete_object(group, "manualModif")
        write(group, "manualModif", manualModif)
        dset = group["manualModif"]
        write_attribute(dset, "dataType", "manualModif")
        write_attribute(dset, "min", minimum(manualModif))
        write_attribute(dset, "max", max(maximum(manualModif), 1))

    end#if


end#saveManualModif


"""
fid - the reference object to hdf5
patienGroupName - name of the group (by hdf5 terminology) in which we want to put our data
mainScrollDat - the object holding all required data for displaying including arrays data
name - name that will identify this array
dataType - wheather it is CT scan ... 
"""
function saveMaskbyName(fid,patienGroupName , mainScrollDat, name, dataType=name)
    arr= filter((it)->it.name==name ,mainScrollDat.dataToScroll)[1].dat
    group = fid[patienGroupName]
    if(!haskey(group, name))
        write(group, name, arr)
        dset = group[name]
        write_attribute(dset, "dataType", dataType)
        write_attribute(dset, "min", minimum(arr))
        write_attribute(dset, "max", max(maximum(arr), 1))

    else
        delete_object(group, name)
        write(group, name, arr)
        dset = group[name]
        write_attribute(dset, "dataType", dataType)
        write_attribute(dset, "min", minimum(arr))
        write_attribute(dset, "max", max(maximum(arr), 1))

    end#if


end#saveMaskbyName 


end #HDF5saveUtils