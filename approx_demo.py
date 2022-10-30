'''
There are different algorithms that solves the steiner problem
'''
import time
import matplotlib.pyplot as plt
import networkx as nx
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
import numpy as np

# STEIN_FILE = "data/B/b02.stp"
# STEIN_FILE = "data/test.std"


def print_graph(graph, terms=None, sol=None):
    '''
    This function draws a graph in a new window,
    @params :
    - graph : The graph to plot
    - terms : A list of terminal nodes of our steiner problem
    - sol : A list of edges of our solution as a pair of int list
    '''
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos)

    if terms is not None:
        # Nodes will be in red if they are terminals
        nx.draw_networkx_nodes(graph, pos, nodelist=terms, node_color='r')
    if sol is not None:
        # Edges will be in red if they are in the solution
        nx.draw_networkx_edges(graph, pos, edgelist=sol, edge_color='r')
    plt.show()


# verify if a solution is correct and evaluate it
def eval_sol(graph, terms, sol):
    '''
    Check if a solution is correct for the steiner problem
    then, if it is, evaluates this solution
    @params :
    - graph : A graph
    - terms : A list of terminal nodes
    - sol : A list of edges of our solution as a pair of int list
    @return the cost of sol
    '''
    # We create an empty graph
    graph_sol = nx.Graph()

    # We add the edges of our solution
    for (i, j) in sol:
        graph_sol.add_edge(i, j, weight=graph[i][j]['weight'])

    # We have the edges so we can check if it is a tree
    if not nx.is_tree(graph_sol):
        print("Error: the proposed solution is not a tree")
        return -1

    # We check if the terminals are indeed covered
    for i in terms:
        if i not in graph_sol:
            print("Error: a terminal is missing from the solution")
            return -1

    # We compute the size of the solution : It is the sum of all the edges
    cost = graph_sol.size(weight='weight')
    return cost


def eval_sol_bis(graph, terms, sol):
    '''
    This evaluation is slightly different from eval_sol
    because it computes even if the solution is not correct
    but it will add an arbitrary malus to the cost
    @params :
    - graph : A graph
    - terms : A list of terminal nodes
    - sol : A list of edges of our solution as a boolean list
    @return the cost of sol
    '''

    cost = 0
    graph_sol = nx.Graph()
    # As it is a boolean list, we need to convert it as a pair of int list
    for _, pos in enumerate(sol):
        # if the value in our sol is True
        if sol[pos]:
            # Then we get the corresponding edge in our graph
            i, j = list(graph.edges())[pos]
            # and we add it
            graph_sol.add_edge(i, j,
                               weight=graph[i][j]['weight'])

    # If there are multiple connected components, we add a malus
    cost += 200*(nx.number_connected_components(graph_sol)-1)

    # And if all terminals are not covered, we add a malus as well
    for i in terms:
        if i not in graph_sol:
            cost += 50

    # We add the weights of all edges
    cost += graph_sol.size(weight='weight')
    return cost


def eval_sol_sommet(graph, terms, sol):
    '''
    This evaluation is slightly different from eval_sol
    because it computes even if the solution is not correct
    but it will add an arbitrary malus to the cost
    @params :
    - graph : A graph
    - terms : A list of terminal nodes
    - sol : A list of nodes of our solution as a boolean list
    @return the cost of sol
    '''

    cost = 0

    # We convert our boolean array to a list of int
    nodes = [i+1 for i in range(len(sol)) if sol[i]]

    # If a terminal is not covered we add a malus
    for i in terms:
        if i not in nodes:
            cost += 50

    # We extract a subgraph with our nodes
    my_graph = graph.subgraph(nodes)
    # If there are multiple connected components we add a malus
    cost += 200*(nx.number_connected_components(my_graph)-1)

    # We extract the minimum spanning tree
    my_graph = nx.minimum_spanning_tree(my_graph)

    # We add the weights of all edges
    cost += my_graph.size(weight='weight')
    return cost


def approx_steiner(graph, terms):
    '''
    Computes an approximate solution to the steiner problem
    @params :
    - graph : A graph
    - terms : A list of terminal nodes
    @return A list of edges giving an exact solution of the steiner problem
    '''

    # Getting all paths and distances from our graph
    dij = dict(nx.all_pairs_dijkstra(graph))
    # We create a new graph ac_min
    ac_min = nx.Graph()

    # To avoid doing twice the same path, we save who we
    # already have visited
    not_visited = terms.copy()
    # res will contain our result
    res = []

    # in which we will add edges between two terminal nodes
    for node1 in terms:
        # we now have visited this node
        not_visited.remove(node1)
        # and we need to connect node1 to every other terminals
        for node2 in not_visited:
            ac_min.add_edge(node1, node2, weight=dij[node1][0][node2])

    # we get the minimum spanning tree of this graph
    # this will be the edges we want to decompose
    ac_min = nx.minimum_spanning_tree(ac_min)
    # for each edge of the tree, we will add the whole
    # path to our result
    for (i, j) in ac_min.edges():
        # k_dep is the previous node, we need it to build the path
        # at first, it is just the first node (we can at the same
        # time delete it from our list)
        previous_step = dij[i][1][j].pop(0)
        # and k will follow the path
        for step in dij[i][1][j]:
            # We add each edge of the path to our result
            # Note : (k, k_dep) can't be in the edges because
            # k > k_dep (thanks to the not_visited trick)
            if (previous_step, step) not in res:
                res.append((previous_step, step))
            # the previous node is now k
            previous_step = step
    return res


# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):
    '''
    A class to create a steiner problem instance
    '''
    # notre graphe
    my_graph = nx.Graph()
    terms = []

    # nos objectifs
    def terminals__t(self, line, converted_token):
        '''
        terminal nodes
        '''
        self.terms.append(converted_token[0])

    def graph__e(self, line, converted_token):
        '''
        edges and their weights
        '''
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start, e_end, weight=weight)


def combination(sol1, sol2):
    '''
    Functions that computes a combination of sol1 and sol2
    @params:
    - sol1 : A boolean list (either nodes or edges)
    - sol2 : A boolean list (either nodes or edges)
    '''

    # We select two random bounds
    first_slice = np.random.randint(1, len(sol1)-1)
    second_slice = np.random.randint(first_slice+1, len(sol1))
    # We have n2 > n

    # We select the three parts
    res = sol1.copy()[0:first_slice]
    res2 = sol2.copy()[first_slice:second_slice]
    res3 = sol1.copy()[second_slice:len(sol1)]

    # We put everything in res
    res = np.append(res, res2)
    res = np.append(res, res3)

    return res.tolist()


def algo_recuit(graph, terms, temperature=1, nb_iter=1000,
                type="linear", nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
    simulated annehaling using edges
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - temperature : the starting temperature
    - type : either "linear" or "exponential", the evolution of temperature
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''

    if type == "linear":
        # If it is linear, the evolution will be so
        seuil = temperature/nb_iter
        limit = 0
    elif type == "exponential":
        # Else, we have a coefficient of evolution
        coeff = 0.999
        limit = temperature*np.power(coeff, nb_iter)

    # Our working solution
    our_sol = init_sol(graph, terms, random=True)
    # Its cost
    our_cost = eval_sol_bis(graph, terms, our_sol)
    # Best solution we found
    best_sol = our_sol.copy()
    # Its cost
    best_cost = our_cost

    # this variable will be used for slicing our results (for plots)
    slice_i = 0
    slices = []

    while temperature > limit:
        # We get a neighbour
        current_sol = voisinage(our_sol)
        # and its cost
        cost = eval_sol_bis(graph, terms, current_sol)

        # if it is a local minimum
        if cost <= our_cost:

            # if it is a global minimum
            if cost < best_cost:
                best_cost = cost
                best_sol = current_sol.copy()

            our_sol = current_sol.copy()
            our_cost = cost

        # else, we still have a probability of
        # setting it as our working solution
        else:
            # the probabilty depends on the temperature
            proba = np.exp(-(cost-our_cost)/temperature)
            rand = np.random.uniform()
            if rand <= proba:
                our_sol = current_sol.copy()
                our_cost = cost

        # we modify the temperature
        if type == "linear":
            temperature -= seuil
        elif type == "exponential":
            temperature *= coeff

        # slicing for plots
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            slices.append(best_cost)

    return best_cost, best_sol, slices


def algo_recuit_sommet(graph, terms, temperature=1, nb_iter=1000,
                       type="linear", nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
    simulated annehaling using nodes
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - temperature : the starting temperature
    - type : either "linear" or "exponential", the evolution of temperature
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''

    if type == "linear":
        # If it is linear, the evolution will be so
        seuil = temperature/nb_iter
        limit = 0
    elif type == "exponential":
        # Else, we have a coefficient of evolution
        coeff = 0.999
        limit = temperature*np.power(coeff, nb_iter)

    # Our working solution
    our_sol = init_sol_sommet(graph, terms)
    # Its cost
    our_cost = eval_sol_sommet(graph, terms, our_sol)
    # Best solution we found
    best_sol = our_sol.copy()
    # Its cost
    best_cost = our_cost

    # this variable will be used for slicing our results (for plots)
    slice_i = 0
    slices = []

    while temperature > limit:
        # We get a neighbour
        current_sol = voisinage_sommet(our_sol, terms)
        # and its cost
        cost = eval_sol_sommet(graph, terms, current_sol)

        # if it is a local minimum
        if cost <= our_cost:

            # if it is a global minimum
            if cost < best_cost:
                best_cost = cost
                best_sol = current_sol.copy()

            our_sol = current_sol.copy()
            our_cost = cost

        # else, we still have a probability of
        # setting it as our working solution
        else:
            # the probabilty depends on the temperature
            proba = np.exp(-(cost-our_cost)/temperature)
            rand = np.random.uniform()
            if rand <= proba:
                our_sol = current_sol.copy()
                our_cost = cost

        # we modify the temperature
        if type == "linear":
            temperature -= seuil
        elif type == "exponential":
            temperature *= coeff

        # slicing for plots
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            slices.append(best_cost)

    return best_cost, best_sol, slices


def algo_recuit_sommet_tabu(graph, terms, temperature=1, nb_iter=1000,
                            type="linear", nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
    simulated annehaling + a tabu list using nodes
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - temperature : the starting temperature
    - type : either "linear" or "exponential", the evolution of temperature
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''

    if type == "linear":
        # If it is linear, the evolution will be so
        seuil = temperature/nb_iter
        limit = 0
    elif type == "exponential":
        # Else, we have a coefficient of evolution
        limit = temperature*np.power(0.999, nb_iter)

    # Our working solution
    our_sol = init_sol_sommet(graph, terms)
    # Its cost
    our_cost = eval_sol_sommet(graph, terms, our_sol)
    # Best solution we found
    best_sol = our_sol.copy()
    # Its cost
    best_cost = our_cost

    # The tabu list will contain solutions we won't explore anymore
    tabu_list = [our_sol]

    # this variable will be used for slicing our results (for plots)
    slice_i = 0
    slices = []

    while temperature > limit:
        flag = True
        while flag:
            # We get a neighbour
            current_sol = voisinage_sommet(our_sol, terms)
            # and we check if it is not in the tabu list
            if current_sol not in tabu_list:
                flag = False

        # and its cost
        cost = eval_sol_sommet(graph, terms, current_sol)

        # if it is a local minimum
        if cost <= our_cost:

            # if it is a global minimum
            if cost < best_cost:
                best_cost = cost
                best_sol = current_sol.copy()

            our_sol = current_sol.copy()
            our_cost = cost

        # else, we still have a probability of
        # setting it as our working solution
        else:
            # the probabilty depends on the temperature
            proba = np.exp(-(cost-our_cost)/temperature)
            rand = np.random.uniform()
            if rand <= proba:
                our_sol = current_sol.copy()
                our_cost = cost

        # We append it to the tabu list so we don't explore it anymore
        tabu_list.append(current_sol)
        # if the tabu list is too big, we remove some solutions
        if len(tabu_list) > 10:
            tabu_list.pop(0)

        # we modify the temperature
        if type == "linear":
            temperature -= seuil
        elif type == "exponential":
            temperature *= 0.999

        # slicing for plots
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            slices.append(best_cost)

    return best_cost, best_sol, slices


def algo_genetique(graph, terms, nb_enfants=16, nb_iter=2000, nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
     genetic solution using edges
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - nb_enfants : the number of children we generate each round
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''

    # We init the populations with some random people
    popu = [init_sol(graph, terms, random=True)
                  for i in range(nb_enfants)]

    # This will be used for slicing (plots)
    slice_i = 0
    slices = []

    for i in range(nb_iter):

        combinations = 0
        mutants = 0

        # We generate some combinations of the population
        while combinations < nb_enfants:
            # We select two parents and we merge them
            comb = combination(popu[np.random.randint(nb_enfants)],
                               popu[np.random.randint(nb_enfants)])

            # We add it only if it is not already in the population
            # in order to avoid cloning of the same parent
            if comb not in popu:
                popu.append(comb)
                combinations += 1

        # and we generate some mutations of existing parents
        while mutants < nb_enfants:
            # we select one of the best parents, generates a mutation
            voisin = voisinage(popu[np.random.randint(nb_enfants/2)])
            # and we add it if it is not already in the population
            if voisin not in popu:
                popu.append(voisin)
                mutants += 1

        # We sort the population : the first element of population
        #                          is our current best solution
        popu = sorted(popu, key=lambda x: eval_sol_bis(graph, terms, x))

        # We keep the best ones
        temp = popu.copy()[0:int(nb_enfants/2)]
        # and some of the weaker ones
        for _ in range(int(nb_enfants/2)):
            temp.append(popu[np.random.randint(nb_enfants/2, nb_enfants*3)])
        # and we store it as our current population
        popu = temp

        # slicing
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            best_cost = eval_sol_bis(graph, terms, popu[0])
            slices.append(best_cost)

    best_cost = eval_sol_bis(graph, terms, popu[0])
    best_sol = popu[0]
    return best_cost, best_sol, slices

def algo_genetique_sommet(graph, terms, nb_enfants=16, nb_iter=2000,
                          nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a genetic
     solution using nodes
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - nb_enfants : the number of children we generate each round
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''
    # We init the populations with some random people
    popu = [init_sol_sommet(graph, terms)
                  for i in range(nb_enfants)]

    # This will be used for slicing (plots)
    slice_i = 0
    slices = []

    for i in range(nb_iter):

        combinations = 0
        mutants = 0

        # We generate some combinations of the population
        while combinations < nb_enfants:

            # We select two parents and we merge them
            comb = combination(popu[np.random.randint(nb_enfants)],
                               popu[np.random.randint(nb_enfants)])

            # We add it only if it is not already in the population
            # in order to avoid cloning of the same parent
            if comb not in popu:
                popu.append(comb)
                combinations += 1

        # and we generate some mutations of existing parents
        while mutants < nb_enfants:
            # we select one of the best parents, generates a mutation
            voisin = voisinage_sommet(popu[np.random.randint(nb_enfants/2)], terms)
            # and we add it if it is not already in the population
            if voisin not in popu:
                popu.append(voisin)
                mutants += 1

        # We sort the population : the first element of population
        #                          is our current best solution
        popu = sorted(popu, key=lambda x: eval_sol_sommet(graph, terms, x))

        # We keep the best ones
        temp = popu.copy()[0:int(nb_enfants/2)]
        # and some of the weaker ones
        for _ in range(int(nb_enfants/2)):
            temp.append(popu[np.random.randint(nb_enfants/2, nb_enfants*3)])
        # and we store it as our current population
        popu = temp

        # slicing
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            best_cost = eval_sol_sommet(graph, terms, popu[0])
            slices.append(best_cost)

    best_cost = eval_sol_sommet(graph, terms, popu[0])
    best_sol = popu[0]
    return best_cost, best_sol, slices


def algo_tabu(graph, terms, nb_iter=2000, nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
    tabu list using edges
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''

    # We init a solution
    sol = init_sol(graph, terms, random=True)
    # Its cost
    cost = eval_sol_bis(graph, terms, sol)
    # Best solution we found
    best_sol = sol.copy()
    # Its cost
    best_cost = cost

    # The tabu list will contain solutions we won't explore anymore
    tabu_list = [sol]

    # this variable will be used for slicing our results (for plots)
    slices = []
    slice_i = 0

    for _ in range(nb_iter):
        flag = True
        while flag:
            # We get a neighbour
            sol = voisinage(best_sol)
            # and we check if it is not in the tabu list
            if sol not in tabu_list:
                flag = False

        # and its cost
        cost = eval_sol_bis(graph, terms, sol)

        # if it is the current minimum
        if cost < best_cost:
            best_cost = cost
            best_sol = sol.copy()

        # We append it to the tabu list so we don't explore it anymore
        tabu_list.append(sol)

        # if the tabu list is too big, we remove some solutions
        if len(tabu_list) > 10:
            tabu_list.pop(0)

        # slicing for plots
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            slices.append(best_cost)

    return best_cost, best_sol, slices


def algo_tabu_sommet(graph, terms, nb_iter=2000, nb_slices=10):
    '''
    An heuristic finding a solution to steiner covering problem using a
    tabu list using nodes
    @params:
    - graph : a graph
    - terms : a list of terminal nodes
    - nb_iter : the number of iterations
    - nb_slices : the number of slices of the result (for plotting)
    @return
    - best_cost : the cost of the best solution we found
    - best_sol : the best solution we found
    - slices : our intermediary results
    '''
    # We init a solution
    sol = init_sol_sommet(graph, terms)
    # Its cost
    cost = eval_sol_sommet(graph, terms, sol)
    # Best solution we found
    best_sol = sol.copy()
    # Its cost
    best_cost = cost

    # The tabu list will contain solutions we won't explore anymore
    tabu_list = [sol]

    # this variable will be used for slicing our results (for plots)
    slices = []
    slice_i = 0

    for _ in range(nb_iter):
        flag = True
        while flag:
            # We get a neighbour
            sol = voisinage_sommet(best_sol, terms)
            # and we check if it is not in the tabu list
            if sol not in tabu_list:
                flag = False

        # and its cost
        cost = eval_sol_sommet(graph, terms, sol)

        # if it is the current minimum
        if cost < best_cost:
            best_cost = cost
            best_sol = sol.copy()

        # We append it to the tabu list so we don't explore it anymore
        tabu_list.append(sol)

        # if the tabu list is too big, we remove some solutions
        if len(tabu_list) > 10:
            tabu_list.pop(0)

        # slicing for plots
        slice_i += 1
        if slice_i % (nb_iter/nb_slices) == 0:
            slices.append(best_cost)

    return best_cost, best_sol, slices


def convert_approx_to_bool(approx, my_graph):
    '''
    A function to convert a list of edges to a boolean list
    @params:
    - approx : A list of edges
    - my_graph : a graph
    @return a boolean list corresponding to approx in my_graph
    '''
    # We initialize an array of zeros
    vals = np.zeros(len(my_graph.edges()))

    # For each edge in the approx
    for i in approx:
        # If it is in the graph
        if i in list(my_graph.edges()):
            vals[list(my_graph.edges()).index(i)] = 1
        # else it means that the reverse pair is in the graph
        else:
            j, k = i
            vals[list(my_graph.edges()).index((k, j))] = 1
    return vals

def init_sol(my_graph, terms, random=False):
    '''
    A function to init a solution as a boolean list
    @params:
    - my_graph : a graph
    - terms : terminal nodes
    - random : a boolean
    @return a boolean list corresponding of a list of edges
    '''
    if random:
        # we initialize an array of values between 0 and 1
        return np.random.randint(2, size=(len(my_graph.edges()))).tolist()

    # We get the steiner approximation
    approx = approx_steiner(my_graph, terms)
    # and convert it as a boolean list
    vals = convert_approx_to_bool(approx, my_graph)
    return vals


def init_sol_sommet(my_graph, terms):
    '''
    A function to init a solution as a boolean list
    @params:
    - my_graph : a graph
    - terms : terminal node
    @return a boolean list corresponding to a list of nodes
    '''
    # we initialize an array of values between 0 and 1
    sol = np.random.randint(2, size=(len(my_graph))).tolist()

    # we keep every terminal node in the solution
    for node in terms:
        sol[node-1] = 1
    return sol


def voisinage(sol):
    '''
    A function to generate a neighbour of sol
    @params:
    - sol : a solution as a boolean list corresponding to a list of edges
    @return a boolean list corresponding of a list of edges
    '''

    # We change a random bit
    random_n = int(np.random.randint(0, len(sol)))
    temporary_sol = sol.copy()
    temporary_sol[random_n] = (temporary_sol[random_n]+1) % 2

    return temporary_sol


def voisinage_sommet(sol, terms):
    '''
    A function to generate a neighbour of sol
    @params:
    - sol : a solution as a boolean list corresponding to a list of nodes
    @return a boolean list corresponding of a list of nodes
    '''

    # We change a random bit
    random_n = int(np.random.randint(0, len(sol)))
    temporary_sol = sol.copy()
    # if it is not a terminal node
    if (random_n+1) not in terms:
        temporary_sol[random_n] = (temporary_sol[random_n]+1) % 2

    return temporary_sol


if __name__ == "__main__":
    start = time.perf_counter()
    with open("results2.txt", "w", encoding="utf8") as file:
        for i in range(1, 19):
            STEIN_FILE = ""
            if i < 10:
                STEIN_FILE = "data/B/b0" + str(i) + ".stp"
            elif i < 19:
                STEIN_FILE = "data/B/b" + str(i) + ".stp"
            elif i < 29:
                STEIN_FILE = "data/C/c0" + str(i-18) + ".stp"
            elif i < 39:
                STEIN_FILE = "data/C/c" + str(i-18) + ".stp"

            my_class = MySteinlibInstance()

            with open(STEIN_FILE, encoding="utf8") as my_file:
                my_parser = SteinlibParser(my_file, my_class)
                my_parser.parse()
                # terminaux

                terms = my_class.terms
                # le graphe
                graph = nx.Graph()
                graph = my_class.my_graph
                # notre solution steiner approx
                sol = approx_steiner(graph, terms)
                # print_graph(graph,terms,sol)

                # évaluation de la solution
                file.write(STEIN_FILE + " Approx steiner : " +
                           str(eval_sol(graph, terms, sol)) + "\n")

                cost, sol2, slices = algo_tabu(graph, terms, nb_iter=20000)
                file.write(STEIN_FILE + " Algo tabu arêtes : " + str(cost) +
                           " nb_iter = 20000\n")
                cost, sol2, slices = algo_tabu_sommet(graph, terms,
                                                      nb_iter=20000)
                file.write(STEIN_FILE + " Algo tabu sommets : " + str(cost) +
                           " nb_iter = 20000\n")
                cost, sol2, slices = algo_genetique(graph, terms, nb_enfants=16,
                                                    nb_iter=1000)
                file.write(STEIN_FILE + " Algo génétique arêtes:" + str(cost) +
                           " nb_iter = 1000, nb_enfants=16\n")
                cost, sol2, slices = algo_genetique_sommet(graph, terms,
                                                           nb_enfants=16,
                                                           nb_iter=1000)
                file.write(STEIN_FILE + " Algo génétique sommets: " +
                           str(cost) + " nb_iter = 1000, nb_enfants=16\n")
                cost, sol2, slices = algo_recuit(graph, terms, temperature=10,
                                                 nb_iter=20000,
                                                 type="exponential")
                file.write(STEIN_FILE + " Algo recuit arêtes exponential : " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                cost, sol2, slices = algo_recuit(graph, terms, temperature=10,
                                                 nb_iter=20000, type="linear")
                file.write(STEIN_FILE + " Algo recuit arêtes linear : " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                cost, sol2, slices = algo_recuit_sommet(graph, terms,
                                                        temperature=10,
                                                        nb_iter=20000,
                                                        type="exponential")
                file.write(STEIN_FILE + " Algo recuit sommets exp : " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                cost, sol2, slices = algo_recuit_sommet(graph, terms,
                                                        temperature=10,
                                                        nb_iter=20000,
                                                        type="linear")
                file.write(STEIN_FILE + " Algo recuit sommets lin : " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                cost, sol2, slices = algo_recuit_sommet_tabu(graph, terms,
                                                             temperature=10,
                                                             nb_iter=20000,
                                                             type="linear")
                file.write(STEIN_FILE + " Algo recuit sommets lin + tabu: " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                cost, sol2, slices = algo_recuit_sommet_tabu(graph, terms,
                                                             temperature=10,
                                                             nb_iter=20000,
                                                             type="exponential")
                file.write(STEIN_FILE + " Algo recuit sommets exp + tabu: " +
                           str(cost) + " nb_iter = 20000, temperature = 10\n")
                file.write('\n')
                end = time.perf_counter()
                print(STEIN_FILE, end - start)

            terms.clear()
            graph.clear()
        print("C'EST FINI !")
