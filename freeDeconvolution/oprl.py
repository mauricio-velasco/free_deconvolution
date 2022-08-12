import numpy as np

# Get indices of diagonal and super-diagonal for dimension dim
def jacobi_indices( dim ):
    diagonal_indices   = np.array( np.diag_indices( dim ) )
    extra_diag_indices = diagonal_indices.copy()[:, :-1]
    extra_diag_indices[1,:] += 1
    return diagonal_indices, extra_diag_indices

# Input: Array of 2n+1 moments from c_0 to c_{2n}
from scipy.linalg import cholesky as scipy_cholesky
def jacobi_from_moments( mom_array, debug=False ):
    n = int( 0.5*(len(mom_array)-1) )
    mom_matrix = np.zeros( shape=(n+1, n+1) )
    for index in range( n+1 ):
        mom_matrix[index,:] = mom_array[index:(index+n+1)]

    # Cholesky
    try:
        cholesky = scipy_cholesky( mom_matrix, lower=True )
        print("Cholesky passed!")
        print("")
    except np.linalg.LinAlgError as err:
        print( err  )
        print("")

    if debug:
        print( "Cholesky matrix: ")
        print( cholesky )

    mom_count = cholesky.shape[0]
    diag_indices, extra_indices = jacobi_indices( mom_count )
    #print( diag_indices )
    #print( extra_indices )
    diag_cholesky = cholesky.T[diag_indices[0,:] , diag_indices[1,:]]
    extra_diag    = cholesky.T[extra_indices[0,:], extra_indices[1,:]]
    print( "Diagonal of Cholesky")
    print( diag_cholesky )
    print( "Extra-diagonal of Cholesky")
    print( extra_diag )
    print("")

    # Compute Jacobi
    jacobi_b = diag_cholesky[1:]/diag_cholesky[:-1]
    jacobi_a = np.zeros_like( diag_cholesky )[:-1]
    jacobi_a[0] = extra_diag[0]
    #x_over_y = extra_diag[1:]/diag_cholesky[1:-1]
    #jacobi_a[2:-1] = x_over_y[1:]-x_over_y[:-1]
    x_over_y = extra_diag/diag_cholesky[:-1]
    jacobi_a[1:] = x_over_y[1:]-x_over_y[:-1]
    jacobi_b = jacobi_b[:-1]

    return jacobi_a, jacobi_b

def quadrature_from_jacobi( jacobi_a, jacobi_b, debug=False):
    # Form Jacobi matrix
    mom_count = len( jacobi_a )
    assert( len(jacobi_b) == mom_count - 1)
    diagonal_indices, extra_indices = jacobi_indices( mom_count )
    jacobi = np.zeros( shape=(mom_count, mom_count) )
    jacobi[extra_indices[0,:], extra_indices[1,:]] = jacobi_b
    jacobi = jacobi + jacobi.T + np.diag( jacobi_a )
    if debug:
        print( "Jacobi matrix ")
        print( jacobi )
        print( "" )

    eigen, vectors = np.linalg.eig(jacobi)
    cyclic_vector = vectors[0,:]

    return eigen, cyclic_vector*cyclic_vector