import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes as fill
from tqdm import tqdm


def creatmask(img):
    zero_mask = np.zeros(img.shape,dtype=bool)
    nonzero_mask = img != 0
    nonzero_mask = zero_mask|nonzero_mask
    nonzero_mask = fill(nonzero_mask)
    return nonzero_mask

# def creatbbox(mask):
#     mask_voxel_coord = np.where(mask!=0)
#     z,y,x = mask.shape
#     midzidx = int((np.min(mask_voxel_coord[0])+np.max(mask_voxel_coord[0]))/2)
#     midxidx = int((np.min(mask_voxel_coord[1])+np.max(mask_voxel_coord[1]))/2)
#     midyidx = int((np.min(mask_voxel_coord[2])+np.max(mask_voxel_coord[2]))/2)
#     bbox = [[midzidx-56,midzidx+56],[midxidx-112,midxidx+112],[midyidx-112,midyidx+112]] #112*224*224
#     return bbox

def creatbbox(mask):
    mask_voxel_coord = np.where(mask!=0)
    minz = np.min(mask_voxel_coord[0])
    maxz = np.min(mask_voxel_coord[0])
    midzidx = int((minz + maxz)/2)

    minx = np.min(mask_voxel_coord[1])
    maxx = np.max(mask_voxel_coord[1])
    midxidx = int((minx + maxx)/2)

    miny = np.min(mask_voxel_coord[2])
    maxy = np.max(mask_voxel_coord[2])
    midyidx = int((miny + maxy)/2)
    bboxz = 56
    bboxx = 112
    bboxy =112

    if maxz > 112:
        if (midzidx > 56) and (midzidx < maxz - 56):
            bboxz = midzidx
        elif (midzidx >maxz - 56):
            bboxz = maxz - 56

    if maxx > 225:
        if (midxidx > 112) and (midxidx < maxx - 112):
            bboxx = midxidx
        elif (midxidx > maxx - 112):
            bboxx = maxx - 112

    if maxy > 224:
        if (midyidx > 112) and (midyidx < maxy - 112):
            bboxy = midyidx
        elif (midyidx > maxy-112):
            bboxy = maxy-112


    bbox = [[bboxz-56,bboxz+56],[bboxx-112,bboxx+112],[bboxy-112,bboxy+112]] #112*224*224
    return bbox

def creatsmallbbox(mask):
    mask_voxel_coord = np.where(mask!=0)
    minz = np.min(mask_voxel_coord[0])
    maxz = np.min(mask_voxel_coord[0])
    midzidx = int((minz + maxz)/2)

    minx = np.min(mask_voxel_coord[1])
    maxx = np.max(mask_voxel_coord[1])
    midxidx = int((minx + maxx)/2)

    miny = np.min(mask_voxel_coord[2])
    maxy = np.max(mask_voxel_coord[2])
    midyidx = int((miny + maxy)/2)
    bboxz = 16
    bboxx = 112
    bboxy =112

    if maxz > 32:
        if (midzidx > 16) and (midzidx < maxz - 16):
            bboxz = midzidx
        elif (midzidx >maxz - 16):
            bboxz = maxz - 16

    if maxx > 225:
        if (midxidx > 112) and (midxidx < maxx - 112):
            bboxx = midxidx
        elif (midxidx > maxx - 112):
            bboxx = maxx - 112

    if maxy > 224:
        if (midyidx > 112) and (midyidx < maxy - 112):
            bboxy = midyidx
        elif (midyidx > maxy-112):
            bboxy = maxy-112


    bbox = [[bboxz-16,bboxz+16],[bboxx-112,bboxx+112],[bboxy-112,bboxy+112]] #112*224*224
    return bbox

def convert():
    error_count = 0
    error_list = []
    sucessful_count = 0
    succesful_list = []

    shape = []
    data_path = r"F:\all_users\gaohy\data\ddm\raw_data_195\renji_140"
    patients = os.listdir(data_path)
    minx = 224
    miny = 224
    for patient in tqdm(patients):
        files = os.listdir(os.path.join(data_path,patient))
        for file in files:
            if file.startswith('Segmentation') or file.startswith('label'):
                seg_flie = file
        img_path = os.path.join(data_path,patient,patient+'.nii')
        label_path = os.path.join(data_path,patient,seg_flie)
        if os.path.exists(img_path) and os.path.exists(label_path):
            raw_img = sitk.ReadImage(img_path)
            seg_label = sitk.ReadImage(label_path)
            raw_data = sitk.GetArrayFromImage(raw_img)
            shape.append(raw_data.shape)
            label_data = sitk.GetArrayFromImage(seg_label)
            mask = creatmask(label_data)
            # mask_voxel_coord = np.where(mask != 0)
            # z = np.max(mask_voxel_coord[0]) - np.min(mask_voxel_coord[0])
            # x = np.max(mask_voxel_coord[1]) - np.min(mask_voxel_coord[1])
            # y = np.max(mask_voxel_coord[2]) - np.min(mask_voxel_coord[2])
            # countz.append(z)
            # countx.append(x)
            # county.append(y)
            try:
                bbox = creatsmallbbox(mask)
                crop_data = raw_data[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]]
                crop_img = sitk.GetImageFromArray(crop_data)
                dataset_path = r'F:\all_users\gaohy\data\ddm\temp1'
                write_path = os.path.join(dataset_path,patient+'.nii')
                sitk.WriteImage(crop_img,write_path)
                sucessful_count+=1
                succesful_list.append(patient)
            except (IndexError,RuntimeError):
                error_count=error_count+1
                error_list.append(patient)
            continue
        else:
            error_count+=1
            error_list.append(patient)
    convert_list = {'failed_convrt' : error_list, 'sucessful_list' : succesful_list}
    df = pd.DataFrame(dict([(k,pd.Series(v))for k,v in convert_list.items()]))
    #df.to_excel('../../croptestdata.xlsx',index = False)
    df.to_excel('F:/all_users/gaohy/data/ddm/croptestdata.xlsx',index = False)
    print(error_count)
    print(error_list)

if __name__ == '__main__':
    convert()






