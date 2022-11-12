
import random
import numpy
from solution import solution
import time
import image_metric
from skimage import data, io, img_as_ubyte


def PSO(objf, lb, ub, dim, PopSize, iters,image):

    # PSO parameters
    histogram = numpy.histogram(image, bins=range(256))[0].astype(numpy.float)

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("-inf"))

    pBest = numpy.zeros((PopSize, dim))
    gBest = numpy.zeros(dim)

    gBestScore = float("-inf")

    pos = numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(iters)
    psnr = numpy.zeros(iters)
    ssim = numpy.zeros(iters)
    fsim = numpy.zeros(iters)
    ncc = numpy.zeros(iters)
    mse = numpy.zeros(iters)

    ############################################
    print('PSO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, PopSize):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
            # Calculate objective function for each particle
            fitness = objf(pos[i, :], histogram)

            if pBestScore[i] < fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore < fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax

                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore
        e_thresholds = [0]
        e_thresholds.extend(gBest)
        e_thresholds.extend([len(histogram) - 1])
        e_thresholds.sort()
        region = numpy.digitize(image, bins=e_thresholds)
        regions = region.copy()
        for thi in range(len(e_thresholds)-1):
            th1 = int( e_thresholds[thi] + 1)
            th2 = int( e_thresholds[thi + 1])
            regions[region== thi] = int((th1+th2)/2)
        output = img_as_ubyte(regions)
        output = img_as_ubyte(regions)
        psnr[l]=image_metric.PSNR(image,output)
        ssim[l]=image_metric.SSIM(image,output)
        fsim[l]=image_metric.FSIM(image,output)
        # ncc[l]=image_metric.NCC(image,output)
        mse[l]=image_metric.MSE(image,output)

        if l % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(gBestScore)
                ]
            )
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.bestIndividual = gBest
    s.psnr=psnr
    s.ssim=ssim
    s.fsim=fsim
    s.ncc=ncc
    s.mse=mse
    s.thresholds = e_thresholds
    s.optimizer = "PSO"
    s.objfname = objf.__name__

    return s
