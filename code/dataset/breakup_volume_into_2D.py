import os
import argparse

from PIL import Image
import nibabel as nib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blurred_volume_path', type=str)
    parser.add_argument('--gt_volume_dir', type=str)
    parser.add_argument('--output_2D_images_dir', type=str)
    parser.add_argument('--save_every_n_cuts', type=int, default=8)
    args = parser.parse_args()

    # Load blurred volume and ground truth
    img_gt = nib.load(os.path.join(args.gt_volume_dir, os.path.basename(args.blurred_volume_path))).get_fdata()
    img_blurred = nib.load(args.blurred_volume_path).get_fdata()

    # Loop over each of the images and break it into 2D images
    gt_dir = os.path.join(args.output_2D_images_dir, "ground_truth")
    blured_dir = os.path.join(args.output_2D_images_dir, "blurred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(blured_dir, exist_ok=True)

    for depth_x in range(0, img_blurred.shape[0], args.save_every_n_cuts):
        # Store ground truth image
        im_gt = Image.fromarray(img_gt[depth_x, :, :])
        im_gt.convert("L").save(os.path.join(
            gt_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_x_{depth_x}.png")
        )

        # Store blurred image
        im_blurred = Image.fromarray(img_blurred[depth_x, :, :])
        im_blurred.convert("L").save(os.path.join(
                blured_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_x_{depth_x}.png")
        )

    for depth_y in range(0, img_blurred.shape[1], args.save_every_n_cuts):
        # Store ground truth image
        im_gt = Image.fromarray(img_gt[:, depth_y, :])
        im_gt.convert('L').save(os.path.join(
            gt_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_y_{depth_y}.png")
        )

        # Store blurred image
        im_blurred = Image.fromarray(img_blurred[:, depth_y, :])
        im_blurred.convert('L').save(os.path.join(
            blured_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_y_{depth_y}.png")
        )

    for depth_z in range(0, img_blurred.shape[2], args.save_every_n_cuts):
        # Store ground truth image
        im_gt = Image.fromarray(img_gt[:, :, depth_z])
        im_gt.convert('L').save(os.path.join(
            gt_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_z_{depth_z}.png")
        )

        # Store blurred image
        im_blurred = Image.fromarray(img_blurred[:, :, depth_z])
        im_blurred.convert('L').save(os.path.join(
            blured_dir, f"{os.path.basename(args.blurred_volume_path).split('.nii')[0]}_depth_z_{depth_z}.png")
        )
