#Using Jax
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
#Question: Evaluate the differential equation "Uxx+Uxy-2Uyy" at x=3,y=1

#Defining the function U:

def u(x,y):
        return jnp.power(x,2)+jnp.power(y,2)+2*x*y-6*x+3*y

#Computing du/dx(Ux) and d2u/dx2 (Uxx)

Ux = jacrev(u, argnums = 0)
Uxx =jacrev(Ux, argnums = 0)

#Computing d2u/dxdy (Uxy)
Uxy = jacrev(Ux, argnums = 1)

#Computing du/dy (Uy) and d2u/dy2 (Uyy)

Uy = jacrev(u, argnums = 1)
Uyy = jacrev(Uy, argnums = 1)

#Calculating the value

sol = Uxx(3.0, 1.0)+ Uxy(3.0,1.0) - 2*Uyy(3.0,1.0)
print('The value is: ', sol)
