import numpy as np
from torch import randperm
from matplotlib.pyplot import *
from pylab import *
import random
import numpy
import math
from solution import solution
import time

def AHA_L(objf, lb, ub, dim, npop, max_it):

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    pop_pos = np.zeros((npop, dim))
    for i in range(dim):
        pop_pos[:, i] = np.random.rand(npop) * (ub[i] - lb[i]) + lb[i]
    pop_fit = np.zeros(npop)
    Convergence_curve = np.zeros(max_it)
    s = solution()

    # Loop counter
    print('AHA_L is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    for i in range(npop):
        pop_fit[i] = objf(pop_pos[i, :])
    best_f = float('inf')
    best_x = []
    for i in range(npop):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = pop_pos[i, :]
    his_best_fit = np.zeros(max_it)
    visit_table = np.zeros((npop, npop))
    diag_ind = np.diag_indices(npop)
    visit_table[diag_ind] = float('nan')
    for it in range(max_it):
        # Direction
        visit_table[diag_ind] = float('-inf')
        for i in range(npop):
            direct_vector = np.zeros((npop, dim))
            r = np.random.rand()
            # Diagonal flight
            if r < 1 / 3:
                rand_dim = randperm(dim)
                if dim >= 3:
                    rand_num = np.ceil(np.random.rand() * (dim - 2))
                else:
                    rand_num = np.ceil(np.random.rand() * (dim - 1))

                direct_vector[i, rand_dim[:int(rand_num)]] = 1
            # Omnidirectional flight
            elif r > 2 / 3:
                direct_vector[i, :] = 1
            else:
                # Axial flight
                rand_num = ceil(np.random.rand() * (dim - 1))
                direct_vector[i, int(rand_num)] = 1
            # Guided foraging
            if np.random.rand() < 0.5:
                MaxUnvisitedTime = max(visit_table[i, :])
                TargetFoodIndex = visit_table[i, :].argmax()
                MUT_Index = np.where(visit_table[i, :] == MaxUnvisitedTime)
                if len(MUT_Index[0]) > 1:
                    Ind = pop_fit[MUT_Index].argmin()
                    TargetFoodIndex = MUT_Index[0][Ind]
                newPopPos = pop_pos[TargetFoodIndex, :] + np.random.randn() * direct_vector[i, :] * (
                        pop_pos[i, :] - pop_pos[TargetFoodIndex, :])
                for j in range(dim):
                    newPopPos[j] = numpy.clip(newPopPos[j], lb[j], ub[j])
                newPopFit = objf(newPopPos[:])
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
                    visit_table[i, TargetFoodIndex] = 0
            else:
                # Territorial foraging
                newPopPos = pop_pos[i, :] + np.random.randn() * direct_vector[i, :] * pop_pos[i, :]
                for j in range(dim):
                    newPopPos[j] = numpy.clip(newPopPos[j], lb[j], ub[j])
                newPopFit = objf(newPopPos[:])
                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    pop_pos[i, :] = newPopPos + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                    visit_table[i, :] += 1
                    visit_table[:, i] = np.max(visit_table, axis=1) + 1
                    visit_table[i, i] = float('-inf')
                else:
                    visit_table[i, :] += 1
        visit_table[diag_ind] = float('nan')
        # Migration foraging
        if np.mod(it, 2 * npop) == 0:
            visit_table[diag_ind] = float('-inf')
            MigrationIndex = pop_fit.argmax()
            pop_pos[MigrationIndex, :] = np.random.rand(dim) * (np.array(ub) - np.array(lb)) + np.array(lb)+ numpy.multiply(numpy.random.randn(dim), Levy(dim))
            visit_table[MigrationIndex, :] += 1
            visit_table[:, MigrationIndex] = np.max(visit_table, axis=1) + 1
            visit_table[MigrationIndex, MigrationIndex] = float('-inf')
            pop_fit[MigrationIndex] = objf(pop_pos[MigrationIndex, :])
            visit_table[diag_ind] = float('nan')
        for i in range(npop):
            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = pop_pos[i, :]

        Convergence_curve[it] = best_f

        if it % 1 == 0:
            print(
                ["At iteration " + str(it) + " the best fitness is " + str(best_f)]
            )
    print(len(Convergence_curve))
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "AHA_L"
    s.objfname = objf.__name__
    s.best = best_f
    s.bestIndividual = best_x 
    return s

def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step
