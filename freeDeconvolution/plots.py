import numpy as np

def plot_population( population_spectrum, num_bins, interval_cdf, population_cdf, plt ):
    # Histogram of population spectrum
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(population_spectrum, num_bins, density=True)
    ax.set_xlabel('Eigenvalues')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of population spectrum for p={}'.format( len(population_spectrum) ))
    fig.tight_layout()
    plt.xlim(0,np.max(population_spectrum)+0.5)
    plt.show()

    # CDF
    plt.plot( interval_cdf, population_cdf )
    plt.xlabel('Eigenvalues')
    plt.ylabel('Probability')
    plt.title(r'CDF of population spectrum for p={}'.format( len(population_spectrum) ))
    plt.xlim(0,np.max(population_spectrum)+0.5)
    plt.show()

def plot_observed_spectrum( Scenario, c, diag, num_bins, plt):
    # Histogram of singular values
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(diag, num_bins, density=True)
    if Scenario=='Case1':
        r = (1+np.sqrt(c))**2 #Right end of MP
        l = (1-np.sqrt(c))**2 #Left end of MP
        y = np.sqrt( (r-bins)*(bins-l) )/(2*np.pi*bins*c) # Added extra c. I believe this is the c part of the mass, while (1-c) is a Dirac at zero
        ax.plot(bins, y, '--', linewidth=4)
    ax.set_xlabel('Eigenvalues')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of singular values for p={}'.format( len(diag) ))
    fig.tight_layout()
    plt.xlim(0,np.max(diag)+0.5)
    plt.show()