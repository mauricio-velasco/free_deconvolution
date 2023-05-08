import numpy as np

def sample_wishart( p, N, population_spectrum=None ):
    if population_spectrum is None:
        population_spectrum = np.ones( (p, ) )
    assert( len(population_spectrum) == p)
    #
    G = np.random.normal( size=(p, N) )
    G = np.dot( np.diag( np.sqrt(population_spectrum)) , G)
    W = G.dot( G.T )
    W = W/N
    # diag, U = np.linalg.eig(W) # old, slow
    diag = np.linalg.eigvalsh(W)
    return np.sort( diag )
