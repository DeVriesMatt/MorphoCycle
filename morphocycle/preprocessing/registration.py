import numpy as np
from scipy.ndimage import affine_transform
import yaml
from addict import Dict
import argparse
from pathlib import Path
import tifffile as tfl
from tqdm import tqdm



def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/home/mvries/Documents/GitHub/MorphoCycle/configurations/registration.yaml", type=str
    )
    args = parser.parse_args()
    return args


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def wrap_affine(
    images, scipy_affineMtx_2D, dst_xy_shape, flip_x=False, flip_y=False, **kwargs
):
    affine_mtx = np.diag(np.ones(images.ndim + 1))[:-1, :]
    affine_mtx[-2:, -3:] = np.array(scipy_affineMtx_2D)
    dst_shape = list(images.shape)
    dst_shape[-2:] = dst_xy_shape[-2:]

    flip_ax = []
    if flip_y:
        flip_ax.append(-2)
    if flip_x:
        flip_ax.append(-1)

    transformed_images = affine_transform(
        input=images if not flip_ax else np.flip(images, axis=tuple(flip_ax)),
        matrix=affine_mtx,
        output_shape=tuple(dst_shape),
    )
    return transformed_images


def transform_qpi(cfg):
    print(cfg.Registration.qpi_dir)
    qpi_dir = sorted(Path(cfg.Registration.qpi_dir).glob("*.tif"))

    for qpi_image in tqdm(qpi_dir):
        print(f"Transforming {qpi_image.name}")
        qpi = tfl.imread(qpi_image)
        registered_qpi = wrap_affine(
            qpi,
            cfg.Registration.scipy_affineMtx_2D,
            cfg.Registration.dst_xy_shape,
            flip_x=cfg.Registration.flip_x,
            flip_y=cfg.Registration.flip_y,
        )
        save_dir = Path(cfg.Registration.save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        tfl.imsave(save_dir / qpi_image.name, registered_qpi)

        print(f"Saved {qpi_image.name} to {cfg.Registration.save_dir}")


if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    cfg.config = args.config
    transform_qpi(cfg)
