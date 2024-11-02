from random import random, randint
from runners.default import run_net
import torchvision.transforms as transforms
import webdataset as wds
import torchvision.datasets as datasets
from math import inf
max_iters = 40
max_pop = 20
cross_prob = .5
mut_prob = .01

min_filters = 10
max_filters = 50
epochs = 64

kernels = [2, 3, 5]


def get_kernel():
    index = randint(0, len(kernels) - 1)
    return kernels[index]


def get_filter():
    return randint(min_filters, max_filters)


def crossover(parent_1, parent_2, prob):
    c1 = [*parent_1]
    c2 = [*parent_2]
    print(c1, c2, "crossover, c1 and c2")
    if random() < prob:
        pt = randint(1, len(parent_1) - 1)
        c1 = parent_1[:pt] + parent_2[pt:]
        c2 = parent_2[:pt] + parent_1[pt:]

    return [c1, c2]


def mutate_children(children_1, children_2, prob):
    return [mutate(children_1, prob), mutate(children_2, prob)]


def mutate(gene, prob):
    new_gene = [*gene]
    if random() < prob:
        mutating_index = randint(0, len(new_gene) - 1)
        new_gene[mutating_index] = get_kernel(
        ) if mutating_index > 2 else get_filter()
    return new_gene


def run(train_dataset, test_dataset, base_path='.'):
    population = []
    # initialize population
    for i in range(max_pop):
        gene = [get_filter(), get_filter(), get_filter(),
                get_kernel(), get_kernel(), get_kernel()]
        population.append(gene)

    i = 0
    fits = []
    while i < max_iters:
        print(f'----------------- Iter: {i + 1}')
        fits = []
        i += 1
        for j, gene in enumerate(population):
            print(f'{j} gene: {gene}')
            acc = run_net(gene, train_dataset, test_dataset, epochs, base_path)
            print(f'Fit: ({1-acc}), Acc: {acc}')
            fits.append((1 - acc))

        min_1 = inf
        min_2 = inf
        index_1 = -1
        index_2 = -1
        # select best 2
        for j, fit in enumerate(fits):
            if fit < min_1:
                min_1 = fit
                index_1 = j
                continue
            if fit < min_2:
                min_2 = fit
                index_2 = j
            if min_2 < min_1:
                temp_value = min_1
                temp_index = index_1
                min_1 = min_2
                index_1 = index_2
                min_2 = temp_value
                index_2 = temp_index

        prev_pop = [*population]
        fittest = [prev_pop[index_1], prev_pop[index_2]]
        print(fittest, "fittest")
        population = []
        for j in range(int(max_pop/2)):
            children = crossover(fittest[0], fittest[1], cross_prob)
            print(children, 'children')
            children = mutate_children(children[0], children[1], mut_prob)
            population = population + children

    min_gene_index = inf
    for j, value in enumerate(fits):
        if value < min_gene_index:
            min_gene_index = j

    return population[min_gene_index]
