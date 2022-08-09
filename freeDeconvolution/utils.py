def newton_raphson( f, f_prime, initial, max_iter=10, tol=1e-12):
    z = initial
    k = 0
    while k<max_iter:
        value = f(z)
        derivative_value = f_prime(z)
        z = z - value/derivative_value
        if abs(value)<tol:
            break
        k = k + 1
    return z

# f is assumed decreasing between a<b
def dichotomy( a, b, f, max_iter=10, tol=1e-8):
    k = 0
    while k<max_iter:
        c = (a+b)/2
        value = f(c)
        if value>0:
            a = c
        else:
            b = c
        k = k + 1
    return (a+b)/2