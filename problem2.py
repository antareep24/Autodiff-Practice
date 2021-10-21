# Use JAX to compute partial derivatives
import jax.numpy as jnp
from jax import jacfwd, jacrev

# f(x, y) = x^2 + y^2 + x^2*y^2
def f(x, y):
    return jnp.power(x, 2) + jnp.power(y, 2) + jnp.power(x, 2)*jnp.power(y, 2)

# Use argnums to specify which variable
dfdx = jacrev(f, argnums=0)
dfdy = jacrev(f, argnums=1)

print('df/dx at (1.0, 1.0): ', dfdx(1.0, 1.0))
print('df/dy at (1.0, 1.0): ', dfdy(1.0, 1.0))
print('df/dx at (2.0, 2.0): ', dfdx(2.0, 2.0))
print('df/dy at (2.0, 2.0): ', dfdy(2.0, 2.0))

# d2f/dx^2 = 2 + 2y^2
# Use jacfwd and jacrev to compute the second derivatives
d2fdx2 = jacfwd(jacrev(f, argnums=0), argnums=0)
print('d2f/dx2 at (1.0, 1.0): ', d2fdx2(1.0, 1.0))

#d2fdy2 = ?
#d2fdxy = ?

