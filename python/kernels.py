import math


def quadratic_kernel(r, h):
    return \
        15 * r/h \
        * ( (r/h)**2 / 4 - r/h + 1 ) \
        / ( 16 * math.pi * h**3 )

def d_quadratic_kernel(r, h):
    return \
        15 * ( (r/h)/2 - 1 ) \
        / ( 16 * math.pi * h**4)
