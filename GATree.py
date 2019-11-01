# By Jeff Yuanbo Han (u6617017), 2018-05-30.
import numpy as np
import matplotlib.pyplot as plt
from decisionTree import *

# define GA settings
DNA_SIZE = 22             # number of bits in DNA
POP_SIZE = 50             # population size
DNA_POSITIVE = 5          # max number of positive genes
CROSS_RATE = 0.8          # DNA crossover probability
MUTATION_RATE = 0.05      # mutation probability
N_GENERATIONS = 10        # generation size
depth = 6                 # max depth of Decision Tree


# define objective function (accuracy of Decision Tree)
def F(featuresNo_list):
    accuracy_list = []
    for featuresNo in featuresNo_list:
        dummyX, dummyY = vec_feature(featuresNo, train, headers)
        accuracy_list.append(model(dummyX, dummyY, depth=depth)[1])
    return np.array(accuracy_list)


# define non-zero fitness function for selection
def get_fitness(prediction):
    return prediction + 1e-7 - np.min(prediction)


# covert binary DNA to meaningful list of feature selection numbers
def translateDNA(pop):
    featuresNo_list = []
    for DNA in pop:
        featuresNo_list.append([i+1 for (i,j) in enumerate(DNA) if j == 1])
    return featuresNo_list


# define population select function based on fitness value
# population with higher fitness value has higher chance to be selected
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# define gene crossover function
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # randomly select another individual from population
        i = np.random.randint(0, POP_SIZE, size=1)
        # choose crossover points(bits)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        # produce one child
        parent[cross_points] = pop[i, cross_points]
    return parent


# define mutation function
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# define function to depress redundant positive genes
def depressDNA(child):
    if sum(child) > DNA_POSITIVE:
        ind = [i for (i,j) in enumerate(child) if j == 1]
        np.random.shuffle(ind)
        child[ind[DNA_POSITIVE:]] = 0
    return child


# define initializing function
def initialize(pop_size):
    pop = [1] * DNA_POSITIVE + [0] * (DNA_SIZE - DNA_POSITIVE)
    np.random.shuffle(pop)
    return np.array([pop] * pop_size)


def main():
    print('----------DNA_POSITIVE = {}----------'.format(DNA_POSITIVE))
    # initialise population DNA
    pop = initialize(POP_SIZE)

    # lists for plotting to show learning process
    x = []
    y = []

    for t in range(N_GENERATIONS):
        # convert binary DNA to feature selection numbers
        pop_DNA = translateDNA(pop)
        # compute objective function based on extracted DNA
        F_values = F(pop_DNA)

        # the best population so far
        maximum = np.max(F_values) * 100
        best_DNA = pop[np.argmax(F_values), :]
        print('Generation {}'.format(t+1))
        print("Most fitted DNA: {}".format(best_DNA))
        print("Accuracy = {}%".format(maximum))
        # record for later plotting
        x.append(t+1)
        y.append(maximum)

        # stop in advance if Accuracy = 100%
        if maximum == 100:
            break

        # train GA
        # calculate fitness value
        fitness = get_fitness(F_values)
        # select better population as parent 1
        pop = select(pop, fitness)
        # make another copy as parent 2
        pop_copy = pop.copy()

        for parent in pop:
            # produce a child by crossover operation
            child = crossover(parent, pop_copy)
            # mutate child
            child = mutate(child)
            # lose redundant positive genes
            child = depressDNA(child)
            # in case all genes are 0
            if sum(child) == 0:
                child = initialize(1)[0]
            # replace parent with its child
            parent[:] = child

    # plot final Decision Tree and learning process
    print('----------Final Decision Tree----------')
    dummyX, dummyY, feature_names = vec_feature(translateDNA([best_DNA])[0], train, headers, get_feature_names=True)
    print('Feature List:')
    print(feature_names)
    model(dummyX, dummyY, depth=depth, vis=True)

    plt.scatter(x, y, s=100, lw=0, c='red', alpha=0.5)
    plt.plot(x, y)
    plt.title('{} Positive Genes'.format(DNA_POSITIVE))
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, t+2))
    plt.yticks([80+i*2 for i in range(11)])
    plt.show()


if __name__ == '__main__':
    main()
