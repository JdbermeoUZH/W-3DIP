import os
import argparse

from PIL import Image
import nibabel as nib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str)
    parser.add_argument('--output_2D_images_dir', type=str)
    parser.add_argument('--save_every_n_cuts', type=int, default=8)
    args = parser.parse_args()

    # Load volume
    img = nib.load(args.volume_path).get_fdata()

    # Loop over each of the images and break it into 2D images
    os.makedirs(args.output_2D_images_dir, exist_ok=True)

    for depth_x in range(0, img.shape[0], args.save_every_n_cuts):
        im = Image.fromarray(img[depth_x, :, :])
        im.convert('RGB').save(os.path.join(
            args.output_2D_images_dir,
            f"{os.path.basename(args.volume_path).split('.nii')[0]}_depth_x_{depth_x}.jpeg")
        )

    for depth_y in range(0, img.shape[1], args.save_every_n_cuts):
        im = Image.fromarray(img[:, depth_y, :])
        im.convert('RGB').save(os.path.join(
            args.output_2D_images_dir,
            f"{os.path.basename(args.volume_path).split('.nii')[0]}_depth_y_{depth_y}.jpeg")
        )

    for depth_z in range(0, img.shape[2], args.save_every_n_cuts):
        im = Image.fromarray(img[:, :, depth_z])
        im.convert('RGB').save(os.path.join(
            args.output_2D_images_dir,
            f"{os.path.basename(args.volume_path).split('.nii')[0]}_depth_z_{depth_z}.jpeg")
        )
