import jax.numpy as jnp
from IPython import embed

calib_path = "/media/bryan/shared/kitti2/dataset/sequences/00/calib.txt"


def _load_calib(p):
    with open(p, "r") as f:
        lines = [x.strip() for x in f.readlines()]
        P0 = [float(x) for x in lines[0].split(" ")[1:]]
        P1 = [float(x) for x in lines[1].split(" ")[1:]]

        P0 = jnp.array(P0).reshape((3, 4))
        P1 = jnp.array(P1).reshape((3, 4))

        return P0, P1


if __name__ == "__main__":
    P0, P1 = _load_calib(calib_path)
    embed()