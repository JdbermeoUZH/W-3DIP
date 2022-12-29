import os
from typing import Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import affine_transform, rotate, gaussian_filter
import matplotlib.pyplot as plt

from utils.common_utils import store_volume_nii_gz


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    return (1/kernel.sum()) * kernel


def plot_kernel_log_scale(kernel, x, y, z):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=np.log(kernel), cmap='Greens')
    plt.show()


class Blur3DKernel:
    def __init__(self, kernel_dims_xyz: Tuple[int, int, int]):
        self.shape = np.array(kernel_dims_xyz).astype(int)
        self.xyz_mesh_grid = np.meshgrid(
                  np.linspace(-self.shape[0]/2, self.shape[0]/2, self.shape[0]),
                  np.linspace(-self.shape[1]/2, self.shape[1]/2, self.shape[1]),
                  np.linspace(-self.shape[2]/2, self.shape[2]/2, self.shape[2])
        )

    def create_gaussian_kernel(
            self,
            sigma_xyz: Union[Tuple[float, float, float], np.ndarray] = None,
            center_xyz: Tuple[float, float, float] = (0, 0, 0),
            plot_kernel: bool = False,
            persist_dir: str = None,
            prefix_str: str = ''
    ) -> np.ndarray:
        x, y, z = self.xyz_mesh_grid

        # Need an (N, 3) array of (x, y, z) triple.
        xyz = np.column_stack([x.flat, y.flat, z.flat])
        mu = np.array(center_xyz)

        if sigma_xyz is not None:
            sigma_xyz = np.array(sigma_xyz)
        else:
            sigma_xyz = 0.3 * ((np.array(self.shape) - 1) * 0.5 - 1) + 0.8

        covariance = np.diag(sigma_xyz ** 2)
        kernel = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
        kernel = (1 / kernel.sum()) * kernel        # Normalize kernel to 1
        kernel = kernel.reshape(x.shape)            # Reshape back intended shape

        # Plot the kernel
        if plot_kernel:
            plot_kernel_log_scale(kernel, x, y, z)

        if persist_dir:
            persist_path = os.path.join(
                persist_dir, f"{prefix_str}_gaussian_sigmas_xyz_{'_'.join([str(sigma) for sigma in sigma_xyz])}_" +
                             f"size_{'_'.join(test_kernel.shape.astype(str))}" + ".npy"
            )

            with open(persist_path, 'wb') as f:
                np.save(f, kernel)
                np.save(f, kernel)

        return kernel

    def create_tilted_gaussian(
            self,
            sigma_xyz: Union[Tuple[float, float, float], np.ndarray] = None,
            center_xyz: Tuple[float, float, float] = (0, 0, 0),
            plot_kernel: bool = False,
            persist_dir: str = None,
            prefix_str: str = ''
    ):
        x, y, z = self.xyz_mesh_grid
        gaussian_kernel = self.create_gaussian_kernel(sigma_xyz, center_xyz)
        kernel = rotate(gaussian_kernel, 45, axes=(2, 0), reshape=False)
        kernel = rotate(kernel, 45, axes=(2, 1), reshape=False)
        kernel /= kernel.sum()

        # Plot the kernel
        if plot_kernel:
            plot_kernel_log_scale(kernel, x, y, z)

        if persist_dir:
            persist_path = os.path.join(
                persist_dir, f"tilted_gaussian_sigmas_xyz_size_{'_'.join(test_kernel.shape.astype(str))}" + ".npy"
            )

            with open(persist_path, 'wb') as f:
                np.save(f, kernel)
                np.save(f, kernel)

        return kernel

    def create_dumbbell_kernel_z(
            self,
            lobe_center_frac: float = 7/10,
            plot_kernel: bool = False,
            persist_dir: str = None
    ) -> np.ndarray:

        x, y, z = self.xyz_mesh_grid
        xyz = np.column_stack([x.flat, y.flat, z.flat])

        lobe_center = lobe_center_frac * self.shape[2]/2
        lobes_scale_xy, lobes_scale_z = (1/2) * self.shape[1]/2, (1/2) * (self.shape[2]/2 - lobe_center)
        lobes_covariance = np.diag(np.array([lobes_scale_xy, lobes_scale_xy, lobes_scale_z]) ** 2)

        center_scale_xy, center_scale_z = (1/4) * self.shape[1]/2, (1/3) * lobe_center
        center_covariance = np.diag(np.array([center_scale_xy, center_scale_xy, center_scale_z]) ** 2)

        lobe_1 = multivariate_normal.pdf(xyz, mean=(0, 0, lobe_center), cov=lobes_covariance)
        lobe_1 = _normalize_kernel(lobe_1).reshape(x.shape)
        center = multivariate_normal.pdf(xyz, mean=(0, 0, 0), cov=center_covariance)
        center = _normalize_kernel(center).reshape(x.shape)
        lobe_2 = multivariate_normal.pdf(xyz, mean=(0, 0, -lobe_center), cov=lobes_covariance)
        lobe_2 = _normalize_kernel(lobe_2).reshape(x.shape)

        kernel = (1/5) * (2 * lobe_1 + 2 * lobe_2 + center)

        if plot_kernel:
            plot_kernel_log_scale(kernel, x, y, z)

        if persist_dir:
            persist_path = os.path.join(
                persist_dir, f"dumbbell_size_{'_'.join(test_kernel.shape.astype(str))}" + ".npy"
            )

            with open(persist_path, 'wb') as f:
                np.save(f, kernel)
                np.save(f, kernel)
        return kernel


if __name__ == "__main__":
    kernels = {}
    kernel_dir = os.path.join("..", "..", "..", "data", "kernels")
    os.makedirs(kernel_dir, exist_ok=True)

    # Regular Gaussian kernel, usually dim = 3*sigma or 2*np.ceil(2*sigma + 1)
    kernel_size = (5, 5, 10)
    test_kernel = Blur3DKernel(kernel_dims_xyz=kernel_size)

    regular_gaussian_kernel = test_kernel.create_gaussian_kernel(
        plot_kernel=True,
        sigma_xyz=None,
        persist_dir=kernel_dir,
        prefix_str='regular'
    )
    kernels['regular_gaussian_kernel'] = regular_gaussian_kernel

    # Displaced gaussian kernel
    disp_gaussian_kernel = test_kernel.create_gaussian_kernel(
        plot_kernel=True,
        center_xyz=(0.5, 0.5, 2),
        sigma_xyz=(0.85, 0.85, 1.4),  # sigma_xyz_,
        persist_dir=kernel_dir,
        prefix_str='displaced'
    )
    kernels['displaced_gaussian_kernel'] = disp_gaussian_kernel

    # Broader gaussian Kernel
    broader_gaussian_kernel = test_kernel.create_gaussian_kernel(
        plot_kernel=True,
        sigma_xyz=(1.5, 1.5, 2.25),
        persist_dir=kernel_dir,
        prefix_str='broader'
    )
    kernels['broader_gaussian_kernel'] = broader_gaussian_kernel

    # tilted gaussian, 3*sigma or 2*np.ceil(2*sigma + 1)
    sigma_xyz = 0.3 * ((np.array((3, 3, 6.5)) - 1) * 0.5 - 1) + 0.8
    kernels['tilted_gaussian_kernel'] = test_kernel.create_tilted_gaussian(
        plot_kernel=True,
        sigma_xyz=sigma_xyz,
        persist_dir=kernel_dir,
        prefix_str='tilted'
    )

    # larger kernel
    larger_kernel = Blur3DKernel(kernel_dims_xyz=(10, 10, 20))
    broader_gaussian_kernel = larger_kernel.create_gaussian_kernel(
        plot_kernel=True,
        sigma_xyz=None,
        persist_dir=kernel_dir,
        prefix_str='larger'
    )
    kernels['broader_gaussian_kernel'] = broader_gaussian_kernel

    # Dumbbell kernel
    dumbbell_kernel = test_kernel.create_dumbbell_kernel_z(
        plot_kernel=True,
        persist_dir=kernel_dir
    )
    kernels['dumbbell_kernel'] = dumbbell_kernel

    os.makedirs('kernels', exist_ok=True)

    for kernel_name, kernel_npy in kernels.items():
        print(kernel_name)
        store_volume_nii_gz(
            vol_array=kernel_npy,
            volume_filename=f'{kernel_name}.nii.gz',
            output_dir='./kernels'
        )

        kernel_npy /= np.max(kernel_npy)

        store_volume_nii_gz(
            vol_array=(kernel_npy > 0.1).astype(np.uint8),
            volume_filename=f'{kernel_name}_seg.nii.gz',
            output_dir='./kernels'
        )

