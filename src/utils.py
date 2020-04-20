import numpy as np

# planar hat operator
def hat(xi):
    assert np.shape(xi) == (3,) or np.shape(xi) == (3, 1)
    xi_hat = np.zeros((3, 3))
    xi_hat[:2, :2] = np.array([[0, -xi[2]], [xi[2], 0]])
    xi_hat[:2, 2] = xi[:2]
    return xi_hat

# planar unhat operator
def unhat(xi_hat):
    assert np.shape(xi_hat) == (3, 3)
    xi_hat = np.array(xi_hat)
    xi = np.zeros((3,))
    xi[:2] = xi_hat[:2, 2]
    xi[2] = xi_hat[1, 0]
    return xi

# planar rotation matrix
def rotation(theta):
    assert np.shape(theta) == ()
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R

# planar rigid body transform
def transform(x, y, theta):
    assert np.shape(x) == ()
    assert np.shape(y) == ()
    assert np.shape(theta) == ()
    g = np.eye(3)
    g[:2, :2] = rotation(theta)
    g[:2, 2] = [x, y]
    return g

# converts a twist to a rigid body transform
def twist_to_transform(xi):
    assert np.shape(xi) == (3,) or np.shape(xi) == (3, 1)
    return transform(xi[0], xi[1], xi[2])

# converts a rigid body transform to a twist
def transform_to_twist(g):
    assert np.shape(g) == (3, 3)
    g = np.array(g)
    x = g[0, 2]
    y = g[1, 2]
    theta = np.arctan2(g[1, 0], g[0, 0])
    xi = np.array([x, y, theta])
    return xi

# inverse of a planar rigid body transform
def inverse_transform(g):
    assert np.shape(g) == (3, 3)
    g = np.array(g)
    R = g[:2, :2]
    p = g[:2, 2]
    g_inv = np.eye(3)
    g_inv[:2, :2] = R.T
    g_inv[:2, 2] = -1 * R.T @ p
    return g_inv

# adjoint of a planar rigid body transform
def adjoint_transform(g):
    assert np.shape(g) == (3, 3)
    g = np.array(g)
    R = g[:2, :2]
    p = g[:2, 2]
    ad_g = np.eye(3)
    ad_g[:2, :2] = R
    ad_g[:2, 2] = [p[1], -p[0]]
    return ad_g

# transforms a point
def transform_point(g, p):
    assert np.shape(p) == (2,) or np.shape(p) == (2, 1)
    assert np.shape(g) == (3, 3)
    return (g @ np.append(p, 1))[:-1]

# transforms a vector
def transform_vector(g, v):
    assert np.shape(v) == (2,) or np.shape(v) == (2, 1)
    assert np.shape(g) == (3, 3)
    return (g @ np.append(v, 0))[:-1]

# converts a point from cartesian to polar
def polar(p):
    assert np.shape(p) == (2,) or np.shape(p) == (2, 1)
    x, y = p
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.arctan2(y, x)
    return np.array([r, theta])

# converts a point from polar to cartesian
def cartesian(p):
    assert np.shape(p) == (2,) or np.shape(p) == (2, 1)
    r, theta = p
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x, y])
