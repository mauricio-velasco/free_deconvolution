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

def aggregate_benchmarks( DOEs, methods ):
    import json

    # Build a more convenient data structure for plotting
    available_scenarios = set([ DOE['Scenario'] for DOE in DOEs] )
    available_Ns        = set([ DOE['N'] for DOE in DOEs] )
    print( "Available Scenarios:", available_scenarios )
    print( "Available Ns       :", available_Ns )

    available_results = { Scenario: {
                                        method: {
                                                    N: { 'errors': [], 'timings': [] } for N in available_Ns
                                                } for method in methods
                                                                } for Scenario in available_scenarios
                        }

    # Step 1: Loop over DOEs
    print( "Step 1: Loop over DOEs...")
    for DOE in DOEs:
        N        = DOE['N']
        scenario = DOE['Scenario']
        for method in methods:
            if not method in DOE['results_by_method']:
                continue
            selected_data = DOE['results_by_method'][method]
            available_results[scenario][method][N]['errors'] .append( DOE["results_by_method"][method]["error"]  )
            available_results[scenario][method][N]['timings'].append( DOE["results_by_method"][method]["timing"] )
    # end of
    # print( json.dumps( available_results, indent=4 ) )

    # Step 2: Build statistics
    print( "Step 2: Building statistics...")
    aggregated_results = { Scenario: {
                                        method: {} for method in methods
                                                                } for Scenario in available_scenarios
                        }
    for scenario in available_scenarios:
        for method in methods:
            # Trick for getting N_range sorted
            N_range = available_results[scenario][method].keys()
            N_range = [ N for N in N_range]
            N_range = np.sort( N_range )
            errors   = [ available_results[scenario][method][N]["errors" ] for N in N_range]
            timings  = [ available_results[scenario][method][N]["timings"] for N in N_range]
            lengths = np.array( [ len(x) for x in errors ] )
            if np.any( lengths==0 ):
                continue
            aggregated_results[scenario][method] = {
                'N'           : N_range,
                'errors_mean' : [ np.mean(x) for x in errors],
                'errors_min'  : [ np.min (x) for x in errors],
                'errors_max'  : [ np.max (x) for x in errors],
                'timings_mean': [ np.mean(x) for x in timings],
                'timings_min' : [ np.min (x) for x in timings],
                'timings_max' : [ np.max (x) for x in timings],
            }
        # end for
    # end for

    # Print for fun
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
        
    #print( json.dumps( aggregated_results, cls=NpEncoder, indent=4 ) )

    return aggregated_results

def make_plots( data, methods ):
    import matplotlib.pyplot as plt

    scenarios = np.sort( [scenario for scenario in data] )

    rows  = len(scenarios)
    plt.rcParams["figure.figsize"] = (15, 30)
    fig, axs = plt.subplots(rows, 1)
    index = 0

    for scenario in scenarios:
        colors=dict()
        colors["convex_optim"] ="r"
        colors["subordination"]="b"
        colors["our method"]   ="g"
        # Plot errors
        # ax = axs[index, 0]
        ax = axs[index]
        for method in methods:
            datum = data[scenario][method]
            if not 'N' in datum:
                continue
            ax.fill_between( datum['N'], datum['errors_min'], datum['errors_max'], label=method, alpha=0.1, color=colors[method])
            ax.plot(         datum['N'], datum['errors_mean'], label=method + " mean", color=colors[method])
        ax.set_xlabel("Population size : N" )
        ax.set_ylabel("$W_1$ error" )
        #plt.xscale("log")
        #plt.yscale("log")
        ax.set_title(scenario)
        ax.legend()

        # Plot timings
        # ax = axs[index, 1]
        # for method in methods:
        #     datum = data[scenario][method]
        #     if not 'N' in datum:
        #         continue
        #     ax.fill_between( datum['N'], datum['timings_min'], datum['timings_max'], label=method, alpha=0.1, color=colors[method])
        #     ax.plot(         datum['N'], datum['timings_mean'], label=method + " mean", color=colors[method])
        # #plt.xscale("log")
        # #plt.yscale("log")       
        # ax.set_xlabel("Population size : N" )
        # ax.set_ylabel("CPU time" )
        # ax.set_title(scenario)
        # ax.legend()

        # Increment index
        index += 1
    # end for
    plt.savefig('benchmark.png')
    plt.show()