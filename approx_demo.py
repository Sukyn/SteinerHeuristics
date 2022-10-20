import sys
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
import time
import numpy as np

stein_file = "data/B/b02.stp"
#stein_file = "data/test.std"


# draw a graph in a window
def print_graph(graph,terms=None,sol=None):

    pos=nx.kamada_kawai_layout(graph)

    nx.draw(graph,pos)
    if (not (terms is None)):
        nx.draw_networkx_nodes(graph,pos, nodelist=terms, node_color='r')
    if (not (sol is None)):
        nx.draw_networkx_edges(graph,pos, edgelist=sol, edge_color='r')
    plt.show()
    return


# verify if a solution is correct and evaluate it
def eval_sol(graph,terms,sol):

    graph_sol = nx.Graph()
    for (i,j) in sol:
        graph_sol.add_edge(i,j,weight=graph[i][j]['weight'])

    # is sol a tree
    if (not (nx.is_tree(graph_sol))):
        print ("Error: the proposed solution is not a tree")
        return -1

    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            print ("Error: a terminal is missing from the solution")
            return -1

    # cost of solution
    cost = graph_sol.size(weight='weight')

    return cost

# give the value of a solution with a list of vertices
def eval_sol_sommet(graph, sol):
    nodes = []
    res = 0

    for i in range(len(sol)) :
        if sol[i]==1 :
            nodes.append(i+1)
    my_graph = graph.subgraph(nodes)

    nb_conn = nx.number_connected_components(my_graph)
    if(nb_conn>1) :
        res = sum([graph[u][v]['weight'] for (u,v) in graph.edges()])
    my_graph = nx.minimum_spanning_tree(my_graph)
    return res + my_graph.size(weight='weight')

# compute a approximate solution to the steiner problem
def approx_steiner(graph,terms):
    # getting all paths and distances from our graph
    dij = dict(nx.all_pairs_dijkstra(graph))
    # we create a new graph ac_min
    ac_min = nx.Graph()

    # to avoid doing twice the same path, we save who we
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
            ac_min.add_edge(node1, node2, weight = dij[node1][0][node2])

    # we get the minimum spanning tree of this graph
    # this will be the edges we want to decompose
    ac_min = nx.minimum_spanning_tree(ac_min)
    # for each edge of the tree, we will add the whole
    # path to our result
    for (i,j) in ac_min.edges():
        # k_dep is the previous node, we need it to build the path
        # at first, it is just the first node (we can at the same
        # time delete it from our list)
        k_dep = dij[i][1][j].pop(0)
        # and k will follow the path
        for k in dij[i][1][j]:
            # We add each edge of the path to our result
            # Note : (k, k_dep) can't be in the edges because
            # k > k_dep (thanks to the not_visited trick)
            if ((k_dep, k) not in res):
                res.append((k_dep, k))
            # the previous node is now k
            k_dep = k
    return res

# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):

    # notre graphe
    my_graph = nx.Graph()
    terms = []

    # nos objectifs
    def terminals__t(self, line, converted_token):
        self.terms.append(converted_token[0])


    def graph__e(self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start,e_end,weight=weight)

def eval_sol_bis(graph,terms,sol):

    cost = 0
    graph_sol = nx.Graph()
    for pos in range(len(sol)):
        if sol[pos]:
            l = list(graph.edges())
            i, j = l[pos]

            graph_sol.add_edge(i,j,
                               weight=graph[i][j]['weight'])


    cost += 1000*(nx.number_connected_components(graph_sol)-1)
    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            cost += 10000
    # cost of solution
    cost += graph_sol.size(weight='weight')
    return cost



def cuit():
    pass

def recuit(graph, terms, temperature=1, nb_iter=1000):
    seuil = temperature/nb_iter
    # Solution sur laquelle on travaille
    our_sol = init_sol(graph,terms, random=False)
    # Poids de cette solution
    our_cost = eval_sol_bis(graph,terms,our_sol)
    # Meilleur poids de solution trouvé
    best_cost = our_cost
    # Meilleure solution trouvée
    best_sol = our_sol.copy()
    while temperature > 0 :
        # Fonction de voisinage -> On modifie
        current_sol = voisinage(our_sol)
        # Nouveaux poids
        cost = eval_sol_bis(graph,terms,current_sol)
        # Si il est mieux on le sauvegarde

        # meilleur local
        if (cost <= our_cost):
            # meilleur absolu
            if (cost < best_cost):
                best_cost = cost
                best_sol = current_sol.copy()
            our_sol = current_sol.copy()
            our_cost = cost
        # probas
        else:
            proba = np.exp(-(cost-our_cost)/temperature)
            print(proba)
            rand = np.random.uniform()
            if (rand <= proba):
                our_sol = current_sol.copy()
                our_cost = cost
        temperature -= seuil
        print(best_cost, our_cost, cost)


    return best_cost, best_sol

def combination(sol1, sol2):
    n = np.random.randint(1, len(sol1))
    res = sol1.copy()[0:n]
    res2 = sol2.copy()[n: len(sol1)]
    res = np.append(res, res2)
    return res.tolist()

def algo_genetique(graph, terms, nb_enfants = 16, nb_iter = 2000):
    population = [init_sol(graph, terms, random=True) for i in range(nb_enfants)]
    #population.append(init_sol(graph,terms,random=False))
    #scores = [eval_sol_bis(boy) for boy in population]
    for i in range(nb_iter):
        j = 0
        k = 0
        while (j < nb_enfants):
            n = np.random.randint(nb_enfants)
            m = np.random.randint(nb_enfants)
            comb = combination(population[n], population[m])
            if (comb not in population):
                population.append(comb)
                j += 1
        while (k < nb_enfants):
            voisin = voisinage(population[np.random.randint(nb_enfants/2)])
            if (voisin not in population):
                population.append(voisin)
                k += 1
        population = sorted(population, key=lambda x: eval_sol_bis(graph, terms, x))
        temp = population.copy()[0:int(nb_enfants/2)]
        for n in range(int(nb_enfants/2)):
            s = np.random.randint(nb_enfants/2, nb_enfants*3)
            temp.append(population[s])
        population = temp
        print(eval_sol_bis(graph, terms, population[0]))
    return population[0], eval_sol_bis(graph, terms, population[0])

def init_sol(my_graph, terms, random=False):

    if (random) :
        return np.random.randint(2, size=(len(my_graph.edges()))).tolist()
    else:
        vals = np.zeros(len(my_graph.edges()))

        # ON RECUPERE LA LISTE D'ARETES DE STEINER
        approx = approx_steiner(my_graph,terms)
        # Et on la convertit en liste de booléens
        for i in approx:
            if i in list(my_graph.edges()):
                #print(i)
                vals[list(my_graph.edges()).index(i)] = 1
            else:
                j, k = i
                vals[list(my_graph.edges()).index((k,j))] = 1
        #print(vals)
        return vals
"""
def init_sol_sommet(my_graph, terms, random=false):
    if(random) :
        np.random.randint(2, size=(len(my_graph)))

"""

def voisinage(sol):
    n = np.random.randint(0, len(sol))
    sol2 = sol.copy()
    sol2[n] = (sol2[n]+1)%2
    return sol2
    '''
    sol2 = sol.copy()
    for i in range(len(sol)):
        n = np.random.random()
        if (0.01 > n):
            sol2[i] = (sol[i]+1)%2
    return sol2
    '''







if __name__ == "__main__":
    start = time.perf_counter()
    my_class = MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        # terminaux
        terms = my_class.terms
        # le graphe
        graph = my_class.my_graph
        #print_graph(graph,terms)

        # notre solution steiner approx
        sol=approx_steiner(graph,terms)
        #print_graph(graph,terms,sol)

        # évaluation de la solution
        print(eval_sol(graph,terms,sol))

        print(algo_genetique(graph, terms))

        #cost, sol2 = recuit(graph, terms, temperature=10, nb_iter=20000)
        #print("Notre maxi résultat")
        #print(cost, sol2)
        end = time.perf_counter()
        print(end - start)
