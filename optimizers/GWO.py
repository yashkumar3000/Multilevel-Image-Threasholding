import random
import numpy
import math
from solution import solution
import time
import image_metric
from skimage import data, io, img_as_ubyte


def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter,image):

    histogram = numpy.histogram(image, bins=range(256))[0].astype(numpy.float)

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("-inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("-inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    psnr = numpy.zeros(Max_iter)
    ssim = numpy.zeros(Max_iter)
    fsim = numpy.zeros(Max_iter)
    ncc = numpy.zeros(Max_iter)
    mse = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :], histogram)

            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness < Alpha_score and fitness > Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)

        Convergence_curve[l] = Alpha_score
        e_thresholds = [0]
        e_thresholds.extend(Alpha_pos)
        e_thresholds.extend([len(histogram) - 1])
        e_thresholds.sort()
        region = numpy.digitize(image, bins=e_thresholds)
        regions = region.copy()
        for thi in range(len(e_thresholds)-1):
            th1 = int( e_thresholds[thi] + 1)
            th2 = int( e_thresholds[thi + 1])
            regions[region== thi] = int((th1+th2)/2)
        output = img_as_ubyte(regions)

        psnr[l]=image_metric.PSNR(image,output)
        ssim[l]=image_metric.SSIM(image,output)
        fsim[l]=image_metric.FSIM(image,output)
        mse[l]=image_metric.MSE(image,output)
        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )
    print(len(Convergence_curve))
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.psnr=psnr
    s.ssim=ssim
    s.fsim=fsim
    s.ncc=ncc
    s.mse=mse
    s.optimizer = "GWO"
    s.objfname = objf.__name__
    s.bestIndividual = numpy.sort(Alpha_pos)
    s.thresholds = e_thresholds

    return s

