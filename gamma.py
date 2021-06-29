import numpy as np
import copy

# BASED on the paper 'Obtaining simultaneous solutions of linear subsystems of inequalities and duals'
# by E.Castillo, F.Jubete, R.E.Pruneda, C.Solares
# https://doi.org/10.1016/S0024-3795(01)00500-6

DEBUG = True
SANITY_SIZE = 100

if DEBUG:
    from termcolor import colored


def dimension_check(A):
    n = len(A[0])
    for a in A:
        if not len(a) == n:
            return False
    return True


def printd(param, color='blue', background=None, attrs=None):
    if DEBUG:
        print(colored(param, color, background, attrs))


def show_table(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column):
    print("V[" + str(h) + "]:-----------------------------------------")

    if h < len(A):
        print("a" + str(h) + str(A[h]) + "\t\t#t\t\t#IA0\t\t#IAdash")
    for i in range(0, len(t)):
        if i in IVbar:
            if i == pivot_column:
                print("*v" + str(i) + str(V[h][i]) + "\t" + str(t[i]) + "\t\t" + str(IA0[i]) + "\t\t" + str(IAdash[i]))
            else:
                print(" v" + str(i) + str(V[h][i]) + "\t" + str(t[i]) + "\t\t" + str(IA0[i]) + "\t\t" + str(IAdash[i]))
        else:
            print(" w" + str(i) + str(V[h][i]) + "\t" + str(t[i]) + "\t\t" + str(IA0[i]) + "\t\t" + str(IAdash[i]))
    print("- - - - - ")
    print("|V[" + str(h) + "]|=" + str(len(V[h])))
    print("----------------------------------------------")


def show_final_table(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column):
    print("V[" + str(h) + "]:-----------------------------------------")

    print("FINAL TABLE")
    for i in range(0, len(V[h])):
        if i in IVbar:

            print(" v" + str(i) + str(V[h][i]) + "\t" + "x" + "\t\t" + str(IA0[i]) + "\t\t" + str(IAdash[i]))
        else:
            print(" w" + str(i) + str(V[h][i]) + "\t" + "x" + "\t\t" + str(IA0[i]) + "\t\t" + str(IAdash[i]))

    print("----------------------------------------------")


def init_gamma_algo(A):
    n = len(A[0])  # we are in the n dimensional space
    V = []
    V.append(np.identity(n, dtype=int))
    IVbar = set(range(0, n))  # idx not used for pivotiong so far
    IA0 = dict()
    IAdash = dict()

    for j in range(0, n):
        IA0[j] = set()
        IAdash[j] = set()

    h = 0
    t = [float("NaN")] * n
    pivot_found = float("NaN")
    pivot_column = float("NaN")
    return A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column


def gamma_step1and2(A, n, V, IVbar, IA0, IAdash, h, t):
    t = np.matmul(V[h], A[h])  # in this line we deviate from the pseudocode in the paper
    # V[h-1] vs V[h]
    # but we started with h=0
    # printd(t)
    pivot_column = float("NaN")
    pivot_found = False
    for j in range(0, len(t)):
        if t[j] == 0:
            IA0[j].add(h)
        else:
            if j in IVbar and not pivot_found:
                pivot_column = j
                pivot_found = True

    return n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column


def gamma_process1(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column):
    # printd(pivot_column)

    if t[pivot_column] > 0:
        sign = 1
        for i in range(0, len(V[h][pivot_column])):
            V[h][pivot_column][i] *= -1
    else:
        sign = -1

    for i in range(0, n):
        for j in range(0, len(t)):
            if (not j == pivot_column) and (not t[j] == 0):
                V[h][j][i] = sign * t[pivot_column] * V[h][j][i] + t[j] * V[h][pivot_column][i]
    # If desired, simplify the column vectors of V by dividing each
    # of them by the greater common divisor of the absolute values of all its components.
    for j in range(0, len(V[h])):
        d = np.gcd.reduce(V[h][j])
        V[h][j] = V[h][j] / d

    IAdash[pivot_column].add(h)
    for j in range(0, len(V[h])):
        if not j == pivot_column:
            IA0[j].add(h)
    IVbar.remove(pivot_column)
    return n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column


def gamma_process2(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column):
    for j in range(0, len(t)):
        if t[j] < 0:
            IAdash[j].add(h)
    Iplusminus = set([])
    for i in range(0, len(V[h])):
        if not i in IVbar:
            if not np.dot(V[h][i], A[h]) == 0:
                Iplusminus.add(i)

    printd("Iplusminus: " + str(Iplusminus), 'red')
    if Iplusminus == set():
        return n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column

    for i in Iplusminus:
        for j in Iplusminus:
            if i < j:
                specialSet = IA0[i].intersection(IA0[j]).union(set([h]))
                counterExampleFound = False
                printd("IVbar:" + str(IVbar), 'red')
                for s in range(0, len(V[h])):
                    if s not in IVbar:

                        if specialSet.issubset(IA0[s]):

                            counterExampleFound = True
                if not counterExampleFound:

                    vStar = t[j] * V[h][i] - t[i] * V[h][j]
                    printd("new vec v*: v" + str(i) + ", v" + str(j) + ":" + str(vStar), 'magenta')
                    printd(V[h])
                    V[h] = np.append(V[h], [vStar], 0)
                    printd(V[h])
                    IA0[len(V[h]) - 1] = specialSet

                    IAdash[len(V[h]) - 1] = set([])
                    for k in range(0, h):
                        if np.dot(vStar, A[k]) < 0:
                            IAdash[len(V[h]) - 1].add(k)

    return n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column


def gamma_algo(A):
    if not dimension_check(A):
        printd("ERROR: Generators are of different dimensions")
        return
    printd("generators are all of the same dimension")
    m = len(A)

    A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column = init_gamma_algo(A)


    for h in range(0, m):  # this check is first half of gammaStep6

        # showTable(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column)
        n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column = gamma_step1and2(A, n, V, IVbar, IA0, IAdash, h, t)
        printd("Step 1+2:")
        show_table(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found,
                   pivot_column)  # most similar situation to the example table in the paper (IA0[j] is alrady updated with t[j]=0)
        if pivot_found:  # this decision is gammaStep 3
            # printd("pivot found:" + str(pivot_column))
            n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column = gamma_process1(A, n, V, IVbar, IA0, IAdash, h, t,
                                                                                     pivot_found, pivot_column)
            printd("G-Process I:")
            # showTable(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column)
        else:  # no pivot has been found
            printd("no pivot found")
            n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column = gamma_process2(A, n, V, IVbar, IA0, IAdash, h, t,
                                                                                     pivot_found, pivot_column)
            printd("G-Process II:")
            # showTable(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column)
        V.append(np.copy(V[h]))
        if DEBUG and len(V[h]) > SANITY_SIZE:
            printd("TOO MANY VECTORS", 'red', attrs=['reverse'])
            break

    print("XXXXXXXXXXXXXXXXXXXXXXXX     FINAL XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    show_final_table(A, n, V, IVbar, IA0, IAdash, h + 1, t, pivot_found, pivot_column)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    printd(V[h + 1])
    h = h + 1
    printd("END of GammaAlgo")
    return A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column


###################


def cone_extraction(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column, generatorChoice='ALL'):  # Algo2

    if generatorChoice == 'ALL':
        generatorChoice = []
        for i in range(0, len(A)):
            generatorChoice.append([i, 1])

    IAplus = dict()
    for i in range(len(V[h])):
        IAplus[i] = set(range(0, len(A))).difference(IA0[i].union(IAdash[i]))
    # Step1
    IB = set()
    ICplus = set()
    ICminus = set()


    for i in range(0, len(generatorChoice)):  # iterate over all indexes becasue we use all in A
        if generatorChoice[i][1] == 0:
            IB.add(generatorChoice[i][0])
        else:
            if generatorChoice[i][1] == -1:
                ICminus.add(generatorChoice[i][0])
            else:
                ICplus.add(generatorChoice[i][0])
    printd("IB=" + str(IB) + "; ICplus=" + str(ICplus) + "; ICminus=" + str(ICminus))
    M = []  # span linear space
    N = []  # generators of acute cone
    printd("IB U ICplus U ICminus=" + str(IB.union(ICplus.union(ICminus))), 'red')
    for idx in range(0, len(V[h])):
        if IA0[idx].issuperset(IB.union(ICplus.union(ICminus))):
            M.append(V[h][idx])
        else:
            if IA0[idx].issuperset(IB) and IA0[idx].union(IAdash[idx]).issuperset(ICplus) and IA0[idx].union(
                    IAplus[idx]).issuperset(ICminus):
                N.append(V[h][idx])
            if IA0[idx].issuperset(IB) and IA0[idx].union(IAplus[idx]).issuperset(ICplus) and IA0[idx].union(
                    IAdash[idx]).issuperset(ICminus):
                N.append(-V[h][idx])
    M = np.array(M)
    N = np.array(N)
    printd("M=" + str(M))
    printd("N=" + str(N))

    printd("End of ConeExtraction")
    return A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column, M, N




def minimal_cone_extraction(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column, generatorChoice, Bstar,
                            Cstar):  # Algo3

    printd("Start minimalCone Extraction", 'white', attrs=['reverse'])

    Bstar = np.array(Bstar)
    Cstar = np.array(Cstar)

    printd("Bstar:" + str(Bstar), 'cyan')
    printd("Cstar" + str(Cstar))
    # STEP 1
    JC = set()
    for c in generatorChoice:
        JC.add(c[0])
    printd("JC= " + str(JC), 'red')

    i = 1  # in the paper but unused... ¯\_(ツ)_/¯
    B = []  # I guess there is a typo in the paper... they did Bstar=[] (and they used sets but that is no issue)
    C = []  # same as above

    # STEP 2

    if not len(Bstar) == 0:  # else: go to step 5

        B.append(Bstar[0])
        for idx in range(0, len(V[h])):
            if np.equal(Bstar[0], V[h][idx]).all():
                K = IA0[idx]
                break
        s = 1
        printd("B=" + str(B))
        printd("K=" + str(K), 'yellow')

        # STEP 3+4
        while s < len(Bstar):  # could have been a for loop over s but i stuck closer to the paper
            s = s + 1
            # STEP 4
            for idx in range(0, len(V[h])):
                if np.equal(Bstar[s - 1], V[h][idx]).all():  # s-1 because of index shift
                    K1 = K.intersection(IA0[idx])
                    break
            if not K1 == K:
                K = K1
            B.append(V[h][idx]) #this indentation is not clear in the paper
                                # but min cone of [[0,0,1]] does not work otherwise
            printd("s:" + str(s), 'green')
            printd("K=" + str(K), 'yellow')
            printd("B=" + str(B))

    # STEP 5
    IAbar0 = dict()
    if not len(Cstar) == 0:  # else: step 7
        printd("JC:" + str(JC), 'red')

        for i in range(0, len(Cstar)):
            try:
                idx = V[h].tolist().index(Cstar[i].tolist())
                IAbar0[idx] = JC.intersection(IA0[idx])
                printd("IA0[" + str(idx) + "]:" + str(IA0[idx]) + "->" + str(IAbar0[idx]), 'yellow')
            except ValueError:
                idx = V[h].tolist().index((-Cstar[i]).tolist())
                IAbar0[-idx] = JC.intersection(IA0[idx])
                printd("IA0[" + str(idx) + "]:" + str(IA0[idx]) + "->" + str(IAbar0[-idx]), 'yellow')

        printd("IAbar0:" + str(IAbar0), 'red')

        # STEP 6
        CstarIdx = []
        for i in IAbar0:
            CstarIdx.append(i)

        for i in CstarIdx:
            for j in CstarIdx:
                if not i == j:
                    if i in IAbar0 and j in IAbar0:
                        if IAbar0[i].issuperset(IAbar0[j]):  # prefers small absolute value of an index
                            del IAbar0[j]
        printd("reduced IAbar0:" + str(IAbar0), 'green')
        for i in IAbar0:
            if i >= 0:
                C.append(V[h][i])
            else:
                C.append(-V[h][-i])

    B = np.array(B)
    C = np.array(C)

    # STEP 7
    printd("B:" + str(B))
    printd("C:" + str(C))
    printd("Minimal Cone Extraction DONE", 'blue', attrs=['reverse'])
    return B, C


if __name__ == '__main__':

    '''
    A = [  # Video 'Algorithm to obtain the dual cone of a given cone' https://www.youtube.com/watch?v=i7iBfBNGnUY&list=LL&index=57
            #they use column vectors but here we have row vectors
        [0, 0, 0, 0, -1],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 0, -1],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, -1, 0, 0],
        [1, 2, 1, 1, 0],
        [0, 2, 0, 1, 0]
    ]
    '''

    '''
    A = [[1, 2, 0, 1, 0], [2, 1, -1, 0, 0], [0, 2, 1, 0, 1], [3, 1, -2, 1, -1], [0, 0, 0, -1, 0],
         [0, 0, 0, 0, -1]]  # paper
    '''

    #A=[[1,1,1],[-1,-1,1],[0,0,1]] #CRITICAL EXAMPLE

    A=[[0,0,1]]

    printd("------", 'magenta')
    A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column = gamma_algo(A)
    printd("------", 'magenta')


    generatorChoice = []
    for i in range(0, len(A)):  # generatorChoice is all generators positive
        generatorChoice.append([i, 1])
    #if DEBUG:
        #generatorChoice = [[0, 1], [2, 1], [3, -1]] #example 2
        #generatorChoice = [[0, 0], [1, 0],[2,1],[3,-1]] #example 3
        #generatorChoice = [[0, 1], [2,1],[3,-1]] #example 4
        #generatorChoice = [[0, 1], [2, 1], [4, 1]]  # example 5


    A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column, M, N = cone_extraction(A, n, V, IVbar, IA0, IAdash, h, t,
                                                                                       pivot_found, pivot_column,
                                                                                       generatorChoice)
    printd("-------", 'magenta')

    Bstar = M
    Cstar = N
    minimal_cone_extraction(A, n, V, IVbar, IA0, IAdash, h, t, pivot_found, pivot_column, generatorChoice, Bstar, Cstar)
