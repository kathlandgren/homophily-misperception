import importlib.util
import pathlib
import numpy as np
import pickle
import networkx as nx
import os

# Load sibling modules from src/ by file path so this works when loaded via importlib.util
_src = pathlib.Path(__file__).parent

def _load_src(name):
    spec = importlib.util.spec_from_file_location(name, _src / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

generate_homophilic_graph_symmetric  = _load_src("generate_homophilic_graph_symmetric")
generate_homophilic_graph_asymmetric = _load_src("generate_homophilic_graph_asymmetric")

def degree_fraction(G, minority_nodes):
    """
    Calculate the fraction of total degrees for majority and minority nodes 
    in a graph and return them as percentages.

    Parameters:
    -----------
    G : networkx.Graph
        A NetworkX graph representing the network.
    minority_nodes : set
        A set of nodes in the graph that belong to the minority group.

    Returns:
    --------
    list of float
        A list containing two values:
        - The percentage of total degrees contributed by minority nodes.
        - The percentage of total degrees contributed by majority nodes.
    
    """
    # Initialize fractions for minority (index 0) and majority (index 1)
    frac = [0, 0]
    
    # Iterate through all nodes in the graph
    for i in range(G.number_of_nodes()):
        # Add the degree of the node to the minority or majority group
        if i in minority_nodes:
            frac[0] += G.degree(i)
        else:
            frac[1] += G.degree(i)
    
    # Normalize the fractions by the total degree sum
    frac = frac / np.sum(frac)
    
    # Convert fractions to percentages and return
    return [i * 100 for i in frac]

def rescale_bayes(y, delta, gamma):
    """
    Rescale an opinion estimate using Bayesian updating.

    This function adjusts an initial opinion estimate `y` based on prior beliefs 
    represented by `delta` and uncertainty represented by `gamma`. 
    The rescaling is done using a Bayesian formula to incorporate prior information into the opinion estimate.
    See Guay et al. 2025 Materials and Methods)

    Parameters:
    -----------
    y : float
        The initial opinion estimate (between 0 and 1).
    delta : float
        The mean prior of majority proportion.
    gamma : float
        The uncertainty -- 0 means more uncertainty in current estimate compared to prior, 1 means more certainty.

    Returns:
    --------
    float
        The rescaled opinion estimate after applying Bayesian updating.
    

    """

    # Apply Bayesian updating formula to rescale the opinion estimate
    numerator = delta**(1-gamma) * y**gamma
    denominator = numerator + (1-y)**gamma
    y_rescaled = numerator / denominator
    
    return y_rescaled

def assess_opinion(n, G, minority_nodes, narcissistic=False, bayes=False, delta=0.5, gamma=0.5):
    """
    Assess the local opinion of a node based on the opinions of its neighbors.

    The function calculates the fraction of a node's neighbors that hold 
    the majority opinion, optionally adjusting the estimate to account for 
    the node's own opinion in a "narcissistic" manner.

    Parameters:
    -----------
    n : int
        The node for which the opinion is being assessed.
    G : networkx.Graph
        A NetworkX graph representing the network structure.
    minority_nodes : set
        A set of nodes in the graph that belong to the minority group.
    narcissistic : bool, optional
        If True, the node's own opinion is included in the calculation 
        (treated as an additional neighbor). Defaults to False.

    Returns:
    --------
    float
        The estimated fraction of neighbors (including self, if narcissistic) 
        that hold the majority opinion.

    """
    neighbor_estimate = 0
    neighbor_degree_count = 0

    # Calculate the opinion based on neighbors' majority/minority status
    for i in G.neighbors(n):
        if i in minority_nodes:
            neighb_opinion = 0
        else:
            neighb_opinion = 1
        neighbor_estimate += neighb_opinion

    # Adjust estimate for narcissistic behavior or if node has no neighbors
    if narcissistic or G.degree[n] == 0:
        if n not in minority_nodes:
            neighbor_estimate += 1  # Node contributes its own majority opinion
        neighbor_degree_count = G.degree[n] + 1  # Include self as a "neighbor"
    else:
        neighbor_degree_count = G.degree[n]

    # Calculate and return the fraction of majority opinions
    y = neighbor_estimate / neighbor_degree_count

    # rescale if doing bayesian rescaling
    if bayes:
        y = rescale_bayes(y, delta=delta, gamma=gamma)

    return y



def assess_opinion_weigh_connected(n, G, minority_nodes, narcissistic=False):
    """
    Assess the local opinion of a node based on the degree-weighted opinions of its neighbors.

    This function calculates the fraction of total degree (weighted by node degrees)
    that aligns with the majority opinion among a node's neighbors. Optionally, it can
    include the node's own opinion if `narcissistic` is set to True.

    Parameters:
    -----------
    n : int
        The node in the graph for which the opinion is being assessed.
    G : networkx.Graph
        A NetworkX graph representing the network structure.
    minority_nodes : set
        A set of nodes in the graph that belong to the minority group.
    narcissistic : bool, optional
        If True, the node's own opinion and degree are included in the calculation.
        Defaults to False.

    Returns:
    --------
    float
        The degree-weighted fraction of opinions supporting the majority.
    """
    # Initialize counters for degree-weighted opinion and total degree
    neighbor_estimate = 0
    neighbor_degree_count = 0

    # Loop through all neighbors of the node
    for i in G.neighbors(n):
        # Determine the opinion of the neighbor (0 for minority, 1 for majority)
        if i in minority_nodes:
            neighb_opinion = 0
        else:
            neighb_opinion = 1

        # Add the degree-weighted opinion to the estimate
        neighbor_estimate += G.degree[i] * neighb_opinion
        # Accumulate the total degree of neighbors
        neighbor_degree_count += G.degree[i]

    # If narcissistic, include the node's own degree and opinion
    if narcissistic:
        neighbor_estimate += G.degree[n]  # Node's own majority opinion
        neighbor_degree_count += G.degree[n]  # Node's own degree

    # Calculate and return the degree-weighted fraction of majority opinions
    y = neighbor_estimate / neighbor_degree_count
    return y

def generate_perceived_opinion(G, minority_nodes, media_nodes, narcissistic=False, weigh_connected=False, bayes=False, delta=0.5, gamma=0.5):
    """
    Generate the perceived and true opinions for all non-media nodes in the graph.

    This function calculates both the true opinion and the perceived opinion for
    nodes in the graph. The perceived opinion can either be unweighted or degree-weighted
    based on the `weigh_connected` parameter. Media nodes are excluded from the analysis.

    Parameters:
    -----------
    G : networkx.Graph
        A NetworkX graph representing the network structure.
    minority_nodes : set
        A set of nodes in the graph that belong to the minority group.
    media_nodes : set
        A set of nodes in the graph that represent media entities (excluded from opinion calculation).
    narcissistic : bool, optional
        If True, the node's own opinion is included in the perception calculation.
        Defaults to False.
    weigh_connected : bool, optional
        If True, the perceived opinion is weighted by the degree of neighboring nodes.
        Defaults to False.

    Returns:
    --------
    tuple of list
        - true_opinion (list): A list of true opinions (0 for minority, 1 for majority) for each non-media node.
        - perceived_opinion (list): A list of perceived opinions for each non-media node.

    """

    # Initialize lists to store true and perceived opinions
    perceived_opinion = []
    true_opinion = []

    # Iterate over all nodes in the graph
    for i in range(G.number_of_nodes()):
        if i not in media_nodes:  # Only consider non-media nodes
            # Determine the true opinion: 0 for minority, 1 for majority
            if i in minority_nodes:
                true_opinion.append(0)
            else:
                true_opinion.append(1)

            # Calculate perceived opinion based on weighting preference
            if not weigh_connected:
                # Unweighted perceived opinion
                perceived_opinion.append(assess_opinion(i, G, minority_nodes, narcissistic=narcissistic,bayes=bayes, delta=delta, gamma=gamma))
            else:
                # Degree-weighted perceived opinion
                perceived_opinion.append(assess_opinion_weigh_connected(i, G, minority_nodes, narcissistic=narcissistic))

    # Return both true and perceived opinions as lists
    return true_opinion, perceived_opinion



## swapping to oversample 
def swap_top_maj_opinion(G, minority_nodes):
    """
    Swap the opinions of the highest-degree majority node and the lowest-degree minority node.

    This function oversamples the minority group by identifying the highest-degree majority node
    and converting it to a minority node. Simultaneously, it identifies the lowest-degree minority
    node and converts it to a majority node. Node attributes are updated to reflect these changes.

    Parameters:
    -----------
    G : networkx.Graph
        A NetworkX graph representing the network structure.
    minority_nodes : list
        A list of nodes currently designated as minority nodes.

    Returns:
    --------
    tuple
        - G (networkx.Graph): The updated graph with swapped node attributes.
        - minority_nodes (list): The updated list of minority nodes.
    """
    # Sort nodes by degree in descending order
    degreesorted = sorted(G.degree, key=lambda x: x[1], reverse=True)

    # Flags to track when the highest-degree majority node and lowest-degree minority node are swapped
    flag1 = 0  # For highest-degree majority node
    flag2 = 0  # For lowest-degree minority node
    i = 0      # Index for iterating over majority nodes
    j = -1     # Index for iterating over minority nodes (from the end of the list)

    # Find and swap the highest-degree majority node to the minority group
    while flag1 == 0:
        if degreesorted[i][0] not in minority_nodes:  # Check if the node is in the majority
            flag1 = 1  # Mark the swap as complete
            minority_nodes.append(degreesorted[i][0])  # Add the node to the minority group
            # Update node color attribute to represent the minority group (e.g., "red")
            nx.set_node_attributes(G, {degreesorted[i][0]: "red"}, name="color")
        else:
            i += 1  # Move to the next node

    # Find and swap the lowest-degree minority node to the majority group
    while flag2 == 0:
        if degreesorted[j][0] in minority_nodes:  # Check if the node is in the minority
            flag2 = 1  # Mark the swap as complete
            minority_nodes.remove(degreesorted[j][0])  # Remove the node from the minority group
            # Update node color attribute to represent the majority group (e.g., "blue")
            nx.set_node_attributes(G, {degreesorted[j][0]: "blue"}, name="color")
        else:
            j -= 1  # Move to the next node (from the end of the list)

    # Return the updated graph and list of minority nodes
    return G, minority_nodes



def run_simulation(homophily, m, num_agents, num_sim, minority_fraction, save_frequency=100):
    """
    Run a simulation to study opinion dynamics in a homophilic network.

    This function generates homophilic networks and simulates opinion dynamics
    for a given number of agents and simulations. It computes and periodically
    saves the opinions of both minority and majority agents.

    Parameters:
    -----------
    homophily : float
        The homophily parameter controlling the likelihood of same-group connections 
        in the network (ranges from 0 to 1).
    m : int
        The number of edges each new node connects to when the network is generated.
    num_agents : int
        The total number of agents (nodes) in the network.
    num_sim : int
        The total number of simulations to run.
    minority_fraction : float
        The fraction of agents belonging to the minority group (0 < minority_fraction < 1).
    save_frequency : int, optional
        The frequency (in simulations) at which intermediate results are saved. Defaults to 100.

    Returns:
    --------
    None
        The function saves simulation results to disk and prints progress updates.
    """
    try:
        # Calculate majority fraction from the minority fraction
        majority_fraction = 1 - minority_fraction
        print(f"Process {os.getpid()}: Running with majority_fraction={majority_fraction}, minority_fraction={minority_fraction}")

        # Initialize matrices to store results for majority and minority agents
        maj_agent_matrix = np.zeros((num_sim, int(majority_fraction * num_agents) + 1))
        min_agent_matrix = np.zeros((num_sim, int(minority_fraction * num_agents)))

        # Iterate through each simulation
        for sim_idx in range(num_sim):
            # Generate a homophilic graph with a symmetric BA model
            G, minority_nodes = generate_homophilic_graph_symmetric.homophilic_ba_graph(
                N=num_agents, m=m, minority_fraction=minority_fraction, homophily=homophily
            )

            # Generate true and perceived opinions for nodes
            true_opinion, perceived_opinion = generate_perceived_opinion(
                G, minority_nodes, {}, narcissistic=False, weigh_connected=False
            )

            # Extract perceived opinions for minority and majority agents
            minority_opinion = [perceived_opinion[i] for i in minority_nodes]
            majority_opinion = [
                perceived_opinion[x]
                for x in range(len(perceived_opinion))
                if x not in minority_nodes
            ]

            # Store the percentage opinions in the result matrices
            maj_agent_matrix[sim_idx, :] = [x * 100 for x in majority_opinion]
            min_agent_matrix[sim_idx, :] = [x * 100 for x in minority_opinion]

            # Save intermediate results periodically
            if (sim_idx + 1) % save_frequency == 0 or sim_idx == num_sim - 1:
                file_prefix = f"output/homophily_{homophily}_m_{m}_num_agents_{num_agents}_sim_{sim_idx + 1}"
                with open(f"{file_prefix}_maj.pkl", "wb") as f:
                    pickle.dump(maj_agent_matrix[:sim_idx + 1], f)
                with open(f"{file_prefix}_min.pkl", "wb") as f:
                    pickle.dump(min_agent_matrix[:sim_idx + 1], f)
                print(f"Saved intermediate results at simulation {sim_idx + 1}")

        # Indicate simulation completion
        print(f"Simulation complete for homophily={homophily}, m={m}, num_agents={num_agents}")

    except Exception as e:
        # Catch and raise exceptions for debugging
        print(f"Error in simulation for homophily={homophily}, m={m}, num_agents={num_agents}: {e}")
        raise

def run_simulation_with_swaps(homophily, m, num_agents, num_sim, num_swaps, minority_fraction, save_frequency=100):
    """
    Run a simulation of opinion dynamics in a homophilic network with iterative swaps.

    This function generates a series of networks based on the given parameters, performs
    swaps of opinions between majority and minority nodes, and tracks perceived opinions
    as well as swap statuses over multiple simulations.

    Parameters:
    -----------
    homophily : float
        The homophily parameter controlling the likelihood of same-group connections 
        in the network (ranges from 0 to 1).
    m : int
        The number of edges each new node connects to when the network is generated.
    num_agents : int
        The total number of agents (nodes) in the network.
    num_sim : int
        The total number of simulations to run.
    num_swaps : int
        The number of opinion swaps to perform in each simulation.
    minority_fraction : float
        The fraction of agents belonging to the minority group (0 < minority_fraction < 1).
    save_frequency : int, optional
        The frequency (in simulations) at which intermediate results are saved. Defaults to 100.

    Returns:
    --------
    None
        The function saves simulation results to disk and prints progress updates.
    """
    try:
        # Calculate the majority fraction
        majority_fraction = 1 - minority_fraction
        print(f"Process {os.getpid()}: Running with majority_fraction={majority_fraction}, minority_fraction={minority_fraction}")

        # Initialize matrices to store opinion values and swap statuses
        maj_agent_matrix = np.zeros((num_sim, num_swaps + 1, int(majority_fraction * num_agents) + 1))
        min_agent_matrix = np.zeros((num_sim, num_swaps + 1, int(minority_fraction * num_agents)))
        
        maj_swapped_matrix = np.zeros((num_sim, num_swaps + 1, int(majority_fraction * num_agents) + 1), dtype=bool)
        min_swapped_matrix = np.zeros((num_sim, num_swaps + 1, int(minority_fraction * num_agents)), dtype=bool)

        # Run simulations
        for sim_idx in range(num_sim):
            # Generate the initial homophilic network
            G, minority_nodes = generate_homophilic_graph_symmetric.homophilic_ba_graph(
                N=num_agents, m=m, minority_fraction=minority_fraction, homophily=homophily
            )

            # Identify majority nodes and initialize swap status
            majority_nodes = [x for x in range(num_agents) if x not in minority_nodes]
            swapped_majority = np.zeros(len(majority_nodes), dtype=bool)
            swapped_minority = np.zeros(len(minority_nodes), dtype=bool)

            # Perform swaps and track opinions over iterations
            for swap_idx in range(num_swaps + 1):
                if swap_idx > 0:
                    # Perform a swap and update the network and minority nodes
                    G, minority_nodes, swapped_nodes = swap_top_maj_opinion(G, minority_nodes)

                    # Update swapped status for the affected nodes
                    for node in swapped_nodes:
                        if node in majority_nodes:
                            swapped_majority[majority_nodes.index(node)] = True
                        elif node in minority_nodes:
                            swapped_minority[minority_nodes.index(node)] = True

                # Calculate perceived and true opinions
                true_opinion, perceived_opinion = generate_perceived_opinion(
                    G, minority_nodes, {}, narcissistic=False, weigh_connected=False
                )

                # Extract opinions for majority and minority groups
                minority_opinion = [perceived_opinion[i] for i in minority_nodes]
                majority_opinion = [
                    perceived_opinion[x]
                    for x in range(len(perceived_opinion))
                    if x not in minority_nodes
                ]

                # Store opinion data
                maj_agent_matrix[sim_idx, swap_idx, :] = [x * 100 for x in majority_opinion]
                min_agent_matrix[sim_idx, swap_idx, :] = [x * 100 for x in minority_opinion]

                # Store swap status
                maj_swapped_matrix[sim_idx, swap_idx, :] = swapped_majority
                min_swapped_matrix[sim_idx, swap_idx, :] = swapped_minority

            # Save intermediate results periodically
            if ((sim_idx + 1) % save_frequency == 0 or sim_idx == num_sim - 1):
                file_prefix = f"output/homophily_{homophily}_m_{m}_num_agents_{num_agents}_sim_{sim_idx + 1}"
                
                # Save majority opinion matrix
                with open(f"{file_prefix}_maj.pkl", "wb") as f:
                    pickle.dump(maj_agent_matrix[:sim_idx + 1, :, :], f)
                
                # Save minority opinion matrix
                with open(f"{file_prefix}_min.pkl", "wb") as f:
                    pickle.dump(min_agent_matrix[:sim_idx + 1, :, :], f)
                
                # Save swap statuses
                with open(f"{file_prefix}_maj_swapped.pkl", "wb") as f:
                    pickle.dump(maj_swapped_matrix[:sim_idx + 1, :, :], f)
                with open(f"{file_prefix}_min_swapped.pkl", "wb") as f:
                    pickle.dump(min_swapped_matrix[:sim_idx + 1, :, :], f)

                print(f"Saved intermediate results at simulation {sim_idx + 1}")

        print(f"Simulation complete for homophily={homophily}, m={m}, num_agents={num_agents}")

    except Exception as e:
        # Handle errors during simulation
        print(f"Error in simulation for homophily={homophily}, m={m}, num_agents={num_agents}: {e}")
        raise


def run_simulation_wrapper(args):
    return run_simulation(*args)

def run_simulation_with_swaps_wrapper(args):
    return run_simulation_with_swaps(*args)

def load_simulation_results(input_folder, homophily_values, m_values, num_agents_values, sim_number):
    """
    Load simulation results from a set of files generated for different parameter combinations.

    This function reads pickled data files containing majority and minority opinion matrices
    for simulations with specified homophily, `m`, and number of agents values.

    Parameters:
    -----------
    input_folder : str
        Path to the folder containing the simulation result files.
    homophily_values : list of float
        List of homophily values used in the simulations.
    m_values : list of int
        List of `m` values (number of edges per new node) used in the simulations.
    num_agents_values : list of int
        List of agent counts (number of nodes) used in the simulations.
    sim_number : int
        The specific simulation number to load results for.

    Returns:
    --------
    dict
        A dictionary where keys are tuples `(homophily, m, num_agents)`, and values are dictionaries
        containing the majority and minority opinion matrices:
        - "majority": Majority opinion matrix.
        - "minority": Minority opinion matrix.
    
    Notes:
    ------
    - Prints debug information for the files being searched and loaded.
    - Handles missing files gracefully by skipping them and printing a warning.

    """
    # Initialize a dictionary to store results for each parameter combination
    results = {}

    # Debug: List all files in the output folder
    print("Files in input folder:")
    for file in os.listdir(input_folder):
        print(file)

    # Iterate through all combinations of homophily, m, and num_agents values
    for homophily in homophily_values:
        for m in m_values:
            for num_agents in num_agents_values:
                # Construct file paths for majority and minority opinion matrices
                file_prefix = f"homophily_{homophily}_m_{m}_num_agents_{num_agents}_sim_{sim_number}"
                maj_file_path = os.path.join(input_folder, f"{file_prefix}_maj.pkl")
                min_file_path = os.path.join(input_folder, f"{file_prefix}_min.pkl")

                # Debug: Print the file paths being accessed
                print(f"Looking for majority file: {os.path.abspath(maj_file_path)}")
                print(f"Looking for minority file: {os.path.abspath(min_file_path)}")

                try:
                    # Load majority opinion matrix from the pickle file
                    with open(maj_file_path, "rb") as f:
                        maj_data = pickle.load(f)

                    # Load minority opinion matrix from the pickle file
                    with open(min_file_path, "rb") as f:
                        min_data = pickle.load(f)

                    # Store loaded data in the results dictionary
                    results[(homophily, m, num_agents)] = {
                        "majority": maj_data,
                        "minority": min_data
                    }
                    print(f"Loaded results for homophily={homophily}, m={m}, num_agents={num_agents}, sim={sim_number}")
                
                except FileNotFoundError as e:
                    # Handle missing files gracefully
                    print(f"File not found: {e}")
                except Exception as e:
                    # Handle other errors during file loading
                    print(f"Error loading file for homophily={homophily}, m={m}, num_agents={num_agents}, sim={sim_number}: {e}")
    
    # Return the results dictionary
    return results


def load_simulation_results_with_swaps(output_folder, homophily_values, m_values, num_agents_values, sim_number):
    """
    Load simulation results for a given set of parameter combinations and simulation number.

    This function reads majority and minority opinion matrices from pickled files for different
    parameter combinations and organizes the data by swap index. The results are stored in a 
    dictionary for easy access.

    Parameters:
    -----------
    output_folder : str
        Path to the folder containing the simulation result files.
    homophily_values : list of float
        List of homophily values used in the simulations.
    m_values : list of int
        List of `m` values (number of edges per new node) used in the simulations.
    num_agents_values : list of int
        List of agent counts (number of nodes) used in the simulations.
    sim_number : int
        The specific simulation number to load results for.

    Returns:
    --------
    dict
        A dictionary where keys are tuples `(homophily, m, num_agents, swap_idx)`, and values
        are dictionaries containing:
        - "majority": Majority opinion matrix for the specific swap.
        - "minority": Minority opinion matrix for the specific swap.
    """
    # Initialize a dictionary to store results
    results = {}

    # Debug: List all files in the output folder for reference
    print("Files in output folder:")
    for file in os.listdir(output_folder):
        print(file)

    # Iterate over all combinations of homophily, m, and num_agents values
    for homophily in homophily_values:
        for m in m_values:
            for num_agents in num_agents_values:
                # Construct file paths for majority and minority opinion matrices
                file_prefix = f"homophily_{homophily}_m_{m}_num_agents_{num_agents}_sim_{sim_number}"
                maj_file_path = os.path.join(output_folder, f"{file_prefix}_maj.pkl")
                min_file_path = os.path.join(output_folder, f"{file_prefix}_min.pkl")

                # Debug: Print the file paths being accessed
                print(f"Looking for majority file: {os.path.abspath(maj_file_path)}")
                print(f"Looking for minority file: {os.path.abspath(min_file_path)}")

                try:
                    # Load the majority opinion matrix
                    with open(maj_file_path, "rb") as f:
                        maj_data = pickle.load(f)  # Contains opinions for all swaps

                    # Load the minority opinion matrix
                    with open(min_file_path, "rb") as f:
                        min_data = pickle.load(f)  # Contains opinions for all swaps

                    # Determine the number of swaps from the data shape
                    num_swaps = maj_data.shape[1]  # Assuming shape = (num_sim, num_swaps + 1, num_agents)

                    # Iterate over each swap index to store the results
                    for swap_idx in range(num_swaps):
                        # Extract the data for the current swap index
                        maj_data_swap = maj_data[:, swap_idx, :]
                        min_data_swap = min_data[:, swap_idx, :]

                        # Store the data in the results dictionary with swap_idx as part of the key
                        results[(homophily, m, num_agents, swap_idx)] = {
                            "majority": maj_data_swap,
                            "minority": min_data_swap
                        }
                    print(f"Loaded results for homophily={homophily}, m={m}, num_agents={num_agents}, sim={sim_number}")
                
                except FileNotFoundError as e:
                    # Handle missing files gracefully
                    print(f"File not found: {e}")
                except Exception as e:
                    # Handle other errors during file loading
                    print(f"Error loading file for homophily={homophily}, m={m}, num_agents={num_agents}, sim={sim_number}: {e}")
    
    # Return the dictionary containing all loaded results
    return results


def run_simulation_with_swaps_asymmetric(h_ss, h_oo, m, num_agents, num_sim, num_swaps, minority_fraction, save_frequency=100, output_dir="output"):
    """
    Asymmetric-homophily version of run_simulation_with_swaps.

    Parameters:
    -----------
    h_ss : float
        Within-group homophily for the support/majority group (h_bb in the generator).
    h_oo : float
        Within-group homophily for the oppose/minority group (h_aa in the generator).
    m, num_agents, num_sim, num_swaps, minority_fraction, save_frequency : same as run_simulation_with_swaps.
    output_dir : str
        Directory to write pkl output files.
    """
    # h_ab = cross-group prob from minority; h_ba = cross-group prob from majority
    h_ab = 1 - h_oo
    h_ba = 1 - h_ss

    try:
        majority_fraction = 1 - minority_fraction
        print(f"Process {os.getpid()}: Running asymmetric h_ss={h_ss}, h_oo={h_oo}, m={m}, num_agents={num_agents}")

        maj_agent_matrix = np.zeros((num_sim, num_swaps + 1, int(majority_fraction * num_agents) + 1))
        min_agent_matrix = np.zeros((num_sim, num_swaps + 1, int(minority_fraction * num_agents)))

        for sim_idx in range(num_sim):
            G, minority_nodes = generate_homophilic_graph_asymmetric.homophilic_barabasi_albert_graph_assym(
                N=num_agents, m=m, minority_fraction=minority_fraction, h_ab=h_ab, h_ba=h_ba
            )

            for swap_idx in range(num_swaps + 1):
                if swap_idx > 0:
                    G, minority_nodes = swap_top_maj_opinion(G, minority_nodes)

                true_opinion, perceived_opinion = generate_perceived_opinion(
                    G, minority_nodes, {}, narcissistic=False, weigh_connected=False
                )

                minority_opinion = [perceived_opinion[i] for i in minority_nodes]
                majority_opinion = [perceived_opinion[x] for x in range(len(perceived_opinion)) if x not in minority_nodes]

                maj_agent_matrix[sim_idx, swap_idx, :] = [x * 100 for x in majority_opinion]
                min_agent_matrix[sim_idx, swap_idx, :] = [x * 100 for x in minority_opinion]

            if (sim_idx + 1) % save_frequency == 0 or sim_idx == num_sim - 1:
                h_ss_str = str(h_ss).replace(".", "p")
                h_oo_str = str(h_oo).replace(".", "p")
                file_prefix = f"{output_dir}/hss_{h_ss_str}_hoo_{h_oo_str}_m_{m}_num_agents_{num_agents}_sim_{sim_idx + 1}"

                with open(f"{file_prefix}_maj.pkl", "wb") as f:
                    pickle.dump(maj_agent_matrix[:sim_idx + 1, :, :], f)
                with open(f"{file_prefix}_min.pkl", "wb") as f:
                    pickle.dump(min_agent_matrix[:sim_idx + 1, :, :], f)

                print(f"Saved intermediate results at simulation {sim_idx + 1}")

        print(f"Simulation complete for h_ss={h_ss}, h_oo={h_oo}, m={m}, num_agents={num_agents}")

    except Exception as e:
        print(f"Error in simulation for h_ss={h_ss}, h_oo={h_oo}, m={m}, num_agents={num_agents}: {e}")
        raise


def run_simulation_wrapper_with_swaps_asymmetric(params):
    """Unpacks tuple for ProcessPoolExecutor: (h_ss, h_oo, m, num_agents, num_sim, num_swaps, minority_fraction[, output_dir])"""
    return run_simulation_with_swaps_asymmetric(*params)


def load_simulation_results_with_swaps_asymmetric(output_folder, conditions, m_values, num_agents_values, sim_number):
    """
    Load asymmetric swap simulation results from pkl files.

    Parameters:
    -----------
    output_folder : str
        Path to folder containing pkl files.
    conditions : list of (h_ss, h_oo) tuples
        Asymmetric homophily conditions.
    m_values, num_agents_values, sim_number : same as load_simulation_results_with_swaps.

    Returns:
    --------
    dict keyed by (h_ss, h_oo, m, num_agents, swap_idx)
    """
    results = {}

    for h_ss, h_oo in conditions:
        for m in m_values:
            for num_agents in num_agents_values:
                h_ss_str = str(h_ss).replace(".", "p")
                h_oo_str = str(h_oo).replace(".", "p")
                file_prefix = f"hss_{h_ss_str}_hoo_{h_oo_str}_m_{m}_num_agents_{num_agents}_sim_{sim_number}"
                maj_file_path = os.path.join(output_folder, f"{file_prefix}_maj.pkl")
                min_file_path = os.path.join(output_folder, f"{file_prefix}_min.pkl")

                try:
                    with open(maj_file_path, "rb") as f:
                        maj_data = pickle.load(f)
                    with open(min_file_path, "rb") as f:
                        min_data = pickle.load(f)

                    num_swap_steps = maj_data.shape[1]
                    for swap_idx in range(num_swap_steps):
                        results[(h_ss, h_oo, m, num_agents, swap_idx)] = {
                            "majority": maj_data[:, swap_idx, :],
                            "minority": min_data[:, swap_idx, :],
                        }
                    print(f"Loaded h_ss={h_ss}, h_oo={h_oo}, m={m}, num_agents={num_agents}")

                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                except Exception as e:
                    print(f"Error loading h_ss={h_ss}, h_oo={h_oo}, m={m}, num_agents={num_agents}: {e}")

    return results
