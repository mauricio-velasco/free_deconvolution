import scipy
import numpy as np


scenario2_weights = np.array( [1, 1, 1] )
scenario2_weights = scenario2_weights/np.sum( scenario2_weights )
scenario2_support = np.array( [0.5, 1, 4] )

def initiate_scenario( Scenario, N ):
    # Create population spectrum
    if Scenario=="Case1":
        c = 0.3
        p = int(c*N)

        population_spectrum = np.ones( p )

    elif Scenario=="Case2":
        c = 0.3
        p = int(c*N)
        weights = scenario2_weights
        support = scenario2_support

        population_cdf = np.cumsum( weights )

        population_spectrum = np.zeros( (p,) )
        block_begin = 0
        for i in range( len(weights) ):
            block_end = int( population_cdf[i]*p )
            population_spectrum[block_begin:block_end] = support[i]
            block_begin = block_end

    elif Scenario=="Case3":
        c = 0.2
        p = int(c*N)
        indices = np.arange( 0, p, 1)
        toeplitz_row    = 0.3**indices
        toeplitz = scipy.linalg.toeplitz( toeplitz_row)
        
        population_spectrum, U = np.linalg.eig(toeplitz)
        population_spectrum = np.sort( population_spectrum )
    else:
        print( "Please specify a scenario..." )
        raise ValueError()

    # Population cdf
    interval_max   = np.max(population_spectrum)+0.5
    interval_cdf   = np.linspace( 0, interval_max, 100)
    population_cdf = np.zeros_like( interval_cdf )
    for i in range( len(interval_cdf) ):
        t = interval_cdf[i]
        population_cdf[i] = np.count_nonzero( population_spectrum <= t )
    population_cdf = population_cdf/p

    return c, p, population_spectrum, interval_cdf, population_cdf