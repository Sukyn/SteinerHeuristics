import approx_demo as apr
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
import math
import statistics as stat

if __name__ == "__main__":
    stein_file = sys.argv[1]
    my_class = apr.MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = apr.SteinlibParser(my_file, my_class)
        my_parser.parse()
        # terminaux
        terms = my_class.terms
        # le graphe
        graph = my_class.my_graph


    for T in [0.1,1,10,100]:
        print("Début tempétature " + str(T))
        nb_test = 10
        nb_iter = 1000
        slices= 10

        pas = nb_iter/slices
        
        res = []
        
        moy = [0 for i in range(slices)]

        for j in range(nb_test):
            sol_v,sol,list_sol = apr.recuit(graph,terms, T, nb_iter, slices)
            res.append(list_sol)
            for i in range(slices):
                moy[i] += list_sol[i]
    
        moy = [i/nb_test for i in moy]
        inter = []
        for i in range(number):
            et = stat.stdev(res[i])
            alpha = 2*et/math.sqrt(nb_test)
            inter.append(alpha)

        plt_x = [(i+1)*pas for i in range(slices)]
        plt.errorbar(plt_x,moy,yerr = inter, label="Temp = " + str(T))
    tmp_tab = [apr.eval_sol(graph,terms, apr.approx_steiner(graph,terms)) for i in range(len(plt_x))]
    plt.plot(plt_x, tmp_tab, label = "approx")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Cout")
    plt.legend()
    plt.show()

