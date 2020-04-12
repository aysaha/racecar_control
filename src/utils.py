import numpy as np

# calculates the norm of a vector
def norm(v):
    return np.linalg.norm(v)

# normalizes a vector
def unit(v):
    return v / norm(v)

# planar hat operator
def hat(v):
    assert v.shape == (3,) or v.shape == (3, 1)
    v_hat = np.zeros((3, 3))
    v_hat[:2, :2] = np.array([[0, -v[2]], [v[2], 0]])
    v_hat[:2, 2] = v[:2]
    return v_hat

# planar unhat operator
def unhat(v_hat):
    assert v_hat.shape == (3, 3)
    v = np.zeros((3,))
    v[:2] = v_hat[:2, 2]
    v[2] = v_hat[1, 0]
    return v

# planar rotation matrix
def rotation(theta):
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return R

# planar rigid body transform
def transform(x, y, theta):
    g = np.eye(3)
    g[:2, :2] = transform(theta)
    g[:2, 2] = [x, y]
    return g

# converts a twist to a rigid body transform
def twist_to_transform(xi):
    assert xi.shape == (3,) or xi.shape == (3, 1)
    return rigid_body_transform(xi[0], xi[1], xi[2])

# converts a rigid body transform to a twist
def transform_to_twist(g):
    assert g.shape == (3, 3)
    x = g[0, 2]
    y = g[1, 2]
    theta = np.arctan2(g[1, 0], g[0, 0])
    xi = np.array([x, y, theta])
    return xi

# inverse of a planar rigid body transform
def inv(g):
    assert g.shape == (3, 3)
    R = g[:2, :2]
    p = g[:2, 2]
    g_inv = np.eye(3)
    g_inv[:2, :2] = R.T
    g_inv[:2, 2] = -1 * R.T @ p
    return g_inv

# adjoint of a planar rigid body transform
def adj(g):
    assert g.shape == (3, 3)
    R = g[:2, :2]
    p = g[:2, 2]
    adj_g = np.eye((3, 3))
    adj_g[:2,:2] = R
    adj_g[:2, 2] = [p[1], -p[0]]
    return adj_g

# transforms a point
def transform_point(p, xi):
    assert p.shape == (3,) or p.shape == (3, 1)
    return (twist_to_transform(xi) @ np.append(p, 1))[:-1]

# transforms a vector
def transform_vector(v, xi):
    assert v.shape == (3,) or v.shape == (3, 1)
    return (twist_to_transform(xi) @ np.append(v, 0))[:-1]