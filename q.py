# Quaternion utilies in JAX, since we rely on JAX to autodiff Jacobians for us
# Built on top of `jaxlie` for Lie group operations (SO3 in our case)

# https://pypi.org/project/jaxlie/

from jaxlie import SO3
import jax.numpy as jnp


def _from_axis_angle(v, theta):
    v = jnp.array(v)
    # Rotation of theta (radians) around axis v.
    vx, vy, vz = v
    sint = jnp.sin(theta / 2)
    return SO3(jnp.array([jnp.cos(theta / 2), vx * sint, vy * sint,
                          vz * sint]))


def _from_wxyz(wxyz):
    return SO3(wxyz=jnp.array(wxyz))


def _from_vector(v):
    # v.shape = (3,)
    v = jnp.array(v)
    return SO3(jnp.array([0, *v]))


def _to_vector(q):
    return q.wxyz[1:]


def _rotate(q, v):
    # Rotate v (vector quaternion) by q (rotation quaternion)
    return SO3.multiply(q, SO3.multiply(v, q.inverse()))


if __name__ == "__main__":
    r = _from_axis_angle(
        [0, 0, 1],
        jnp.pi / 2,
    )
    p = _from_vector([0, 1, 0])
    print(_rotate(r, p))

    r = _from_axis_angle(
        [1, 0, 0],
        jnp.pi / 2,
    )
    p = _from_vector([0, 1, 0])
    print(_rotate(r, p))

    r = _from_axis_angle(
        [0, 1, 0],  #
        jnp.pi / 2,
    )
    p = _from_vector([1, 0, 0])
    print(_rotate(r, p))

    # Imagine looking straight down at the vector v (rooted at the origin)
    # We then rotate by theta in the counter-clockwise direction.
