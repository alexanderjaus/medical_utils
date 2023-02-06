import argparse
import os
import json

from PIL import Image

import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Convert nibabel 3D images to slices')
parser.add_argument("source_folder", type=str, help="Path to source folder containing nii.gz files")
parser.add_argument("target_folder", type=str, help="Path to target folder")
parser.add_argument("slice_dim", type=int, help="Axis along which slicing is done")
parser.add_argument("spacing", type=int, help="Spacing in voxels between png slices")
parser.add_argument("global_norm", type=bool, help="Use global dataset mean and variance, instead of local", default=True)

def check_args(args):
    assert os.path.isdir(args.source_folder), f"Source folder {args.source_folder} does not exist"
    source_content = list(filter(lambda x: x.endswith(".nii.gz"), os.listdir(args.source_folder)))
    assert len(source_content) > 0, f"Source folder {args.source_folder} containes no nii.gz images"
    assert os.path.isdir(args.target_folder), f"Target folder {args.target_folder} does not exist"
    assert args.slice_dim in [0,1,2], f"Expect slice_dim to be in [0,1,2], got {args.slice_dim}"
    nib_example = nib.load(os.path.join(args.source_folder, source_content[0]))
    nib_example = nib_example.get_fdata()
    assert args.spacing < nib_example.shape[args.slice_dim]

def calc_global_mean(img_source_path, img_list, clipping_values = (-1024,4000)):
    list_of_means = []
    list_of_variances = []

    if "dataset_properties.json" in os.listdir(img_source_path):
        print(f"Found existing global configuation in path {img_source_path}")
        with open(os.path.join(img_source_path,"dataset_properties.json"),"r") as f:
            dataset_properties = json.load(f)
        global_mean = float(dataset_properties["global_mean"])
        global_std = float(dataset_properties["global_std"])
        return global_mean, global_std

    print("No existing data for the dataset found. Will now estimating global mean of dataset")
    for idx, img in enumerate(img_list):
        if idx%10 == 0:
            print(f"Processing image {idx}/{len(img_list)}")
        voxels = nib.load(os.path.join(img_source_path,img)).get_fdata()
        voxels = np.clip(voxels, clipping_values[0], clipping_values[1])
        list_of_means.append(np.mean(voxels))
        list_of_variances.append(np.var(voxels))
    global_mean = np.mean(list_of_means)
    global_std = np.sqrt(np.sum(list_of_variances)/len(list_of_variances))
    
    print("Finished calcuating global mean and std. Will write to a file")
    dataset_properties = {
        "global_mean": global_mean,
        "global_std": global_std,
        "list_of_means": list_of_means,
        "list_of_variances": list_of_variances
    }
    with open("dataset_properties.json","w") as f:
        json.dump(dataset_properties,f)

    return global_mean, global_std
    


def main(args):
    check_args(args)
    source_folder = args.source_folder
    target_folder = args.target_folder
    slice_dim = args.slice_dim
    spacing = args.spacing
    global_norm = args.global_norm

    global_clipping_values = (-1024,3500)

    img_list = list(filter(lambda x:x.endswith(".nii.gz"), os.listdir(source_folder)))
    if global_norm:
        global_mean, global_std = calc_global_mean(source_folder)

    for image in img_list:
        cur_img_path = os.path.join(source_folder, image)
        cur_img = nib.load(cur_img_path)
        header, voxels  = cur_img.header, cur_img.get_fdata()
        voxels_min, voxels_max = np.min(voxels), np.max(voxels)
        if global_norm:
            voxels = np.clip(voxels,global_clipping_values[0], global_clipping_values[1])
            voxels_converted = ((voxels - global_clipping_values[0]) / (voxels_max-voxels_min))*255
            converted_mean = (global_mean - global_clipping_values[0]) / (voxels_max-voxels_min)*255
            converted_std = global_std / ((global_clipping_values[1]-global_clipping_values[0])/(255))
            voxels_converted = (voxels_converted-converted_mean)/converted_std
        else:
            voxels_converted = ((voxels - voxels_min) / (voxels_max-voxels_min))*255
        target_image_folder = os.path.join(target_folder,image.split(".")[0])
        if not os.path.exists(target_image_folder):
            os.mkdir(target_image_folder)
        for i in range(0, voxels_converted.shape[slice_dim], spacing):
            if slice_dim == 0:
                img_slice = voxels_converted[i,:,:]
            elif slice_dim == 1:
                img_slice = voxels_converted[:,i,:]
            else:
                img_slice = voxels_converted[:,:,i]
            img = Image.fromarray(img_slice.astype(np.uint8),mode="L")
            img.save(os.path.join(target_image_folder,f"slice_{i}_{slice_dim}.png"))
    
    with open(os.path.join(target_folder, "slices_info.json"),"w") as f:
        json.dump(args.__dict__,f)
            


if __name__ == '__main__':
    main(parser.parse_args())