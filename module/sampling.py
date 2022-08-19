import numpy as np
from scipy.stats import laplace, norm
import random


def map_range(li, m_range=(0, 1)):
    li = np.array(li)
    return (li - min(li)) / len(li) * (m_range[1] - m_range[0])

def piecewiseLaplaceSampling(scale=50):
    population_list = [list(range(0, 149)), list(range(149, 249)), list(range(249, 350))]
    size_list = [np.floor(3 / 7 * 50) + 1, np.floor(2 / 7 * 50), np.floor(2 / 7 * 50)]

    samples_list = [0, 1, 2, 3, 4, 149, 150, 151, 152, 153, 249, 250, 251, 252, 253]
    pieces = [0, 149, 249]
    for k, v in enumerate(pieces):
        population = np.array(population_list[k])
        size = np.array(size_list[k])
        population_in_range = map_range(population, (0, 10))
        if k == 0:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale)
        if k == 1:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale) + laplace.pdf(population_in_range, -1490/250, scale)
        if k == 2:
            scaled_pdf = laplace.pdf(population_in_range, 0, scale) + laplace.pdf(population_in_range, -1490/250, scale)\
                         + laplace.pdf(population_in_range, -2490/250, scale)

        normalized_pdf = (scaled_pdf) / sum(scaled_pdf)
        normalized_cmf = np.array([sum(normalized_pdf[:k + 1]) for k, v in enumerate(normalized_pdf)])

        samples = []
        while True:
            _sample = random.random()
            index = 0
            for k, v in enumerate(normalized_cmf):
                if v <= _sample:
                    index = k
                else:
                    break
            if population[index] not in samples_list:
                samples.append(population[index])
            if len(samples) >= size - 5:
                break
        for i in samples:
            samples_list.append(i)

    return np.sort(samples_list)

def laplaceSampling(scale=10):
    population = np.array(list(range(0, 350)))
    size = 50
    population_in_range = map_range(population, (0, 10))
    scaled_pdf = laplace.pdf(population_in_range, 0, scale)

    normalized_pdf = (scaled_pdf) / sum(scaled_pdf)
    normalized_cmf = np.array([sum(normalized_pdf[:k + 1]) for k, v in enumerate(normalized_pdf)])

    samples = []
    while True:
        _sample = random.random()
        index = 0
        for k, v in enumerate(normalized_cmf):
            if v <= _sample:
                index = k
            else:
                break
        if population[index] not in samples:
            samples.append(population[index])
        if len(samples) >= size:
            break

    return np.sort(samples)


def uniformSampling():
    return np.sort([int(i * 350/50) for i in range(50)])

def randomSampling(seed=1):
    random.seed(seed)
    return np.sort(random.sample(range(0, 350), 50))


def main():
    print("piecewiseLaplaceSampling:", piecewiseLaplaceSampling(scale=10))
    print("laplaceSampling2:", laplaceSampling(scale=2))
    print("laplaceSampling5:", laplaceSampling(scale=5))
    print("laplaceSampling10:", laplaceSampling(scale=10))
    print("uniformSampling:", uniformSampling())
    print("randomSampling:", randomSampling(seed=1))

if __name__ == '__main__':
    main()