
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random


# The dataset is uploaded
f = open("Assignment 3 medical_dataset.data")
dataset_X = []
dataset_y = []
line = " "
while line != "":
    line = f.readline()
    line = line[:-1]
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
            else:
                value = float(line[i])
                if value == 0:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        dataset_X.append(floatList)
f.close()

# The dataset is splited into training and test.
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)

# The dataset is scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# The model is created
model = KNeighborsClassifier(n_neighbors = 3)


# Function that calculates the fitness of a solution
def calculateFitness(solution):
    fitness = 0

    # The features are selected according to solution
    X_train_Fea_selc = []
    X_test_Fea_selc = []
    for example in X_train:
        X_train_Fea_selc.append([a*b for a,b in zip(example,solution)])
    for example in X_test:
        X_test_Fea_selc.append([a*b for a,b in zip(example,solution)])

    model.fit(X_train_Fea_selc, y_train)

    # We predict the test cases
    y_pred = model.predict(X_test_Fea_selc)

    # We calculate the Accuracy
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0] # True positives
    FP = cm[0][1] # False positives
    TN = cm[1][1] # True negatives
    FN = cm[1][0] # False negatives

    fitness = (TP + TN) / (TP + TN + FP + FN)

    return round(fitness *100,2)

MAX_FITNESS_CALCULATIONS = 5000


def test():
    solution = [0] * 5
    neighbours = []
    for index, value in enumerate(solution):
        neighbour = solution.copy()
        neighbour[index] = 1 - value
        neighbours.append(neighbour)

    for neighbour in neighbours:
        print(neighbour)

def test2():
    solution = [0] * 5
    neighbours = []
    for index, value in enumerate(solution):
        neighbour = solution.copy()
        neighbour[index] = 1 - value
        neighbours.append(neighbour)

    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i], neighbour[j] = 1 - neighbour[j], 1 - neighbour[i]
            neighbours.append(neighbour)

    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            for k in range(j + 1, len(solution)):
                neighbour = solution.copy()
                neighbour[i], neighbour[j], neighbour[k] = 1 - neighbour[k], 1 - neighbour[j], 1 - neighbour[i]
                neighbours.append(neighbour)

    for neighbour in neighbours:
        print(neighbour)

def hill_climbing():
    global FITNESS_CALCULATIONS_COUNTER
    FITNESS_CALCULATIONS_COUNTER = 0
    randomSolution = []

    for _ in range(0, len(X_train[0])):
        randomSolution.append(random.randint(0, 1))

    currentSolution = randomSolution
    currentSolutionFitness = calculateFitness(currentSolution)
    FITNESS_CALCULATIONS_COUNTER += 1

    bestNeighbour = currentSolution.copy()
    bestNeighbourFitness = currentSolutionFitness


    while True:
        neighbours = []

        for index, value in enumerate(currentSolution):
            neighbour = currentSolution.copy()
            neighbour[index] = 1 - value
            neighbours.append(neighbour)


        for neighbour in neighbours:
            neighbourFitness = calculateFitness(neighbour)
            FITNESS_CALCULATIONS_COUNTER += 1

            if neighbourFitness > bestNeighbourFitness:
                bestNeighbour = neighbour
                bestNeighbourFitness = neighbourFitness

        if bestNeighbourFitness > currentSolutionFitness:
            currentSolution = bestNeighbour
            currentSolutionFitness = bestNeighbourFitness
            print("Best solution fitness: (", FITNESS_CALCULATIONS_COUNTER, "/", MAX_FITNESS_CALCULATIONS, "):", currentSolution, currentSolutionFitness)

        else:
            randomSolution = []
            for _ in range(len(X_train[0])):
                randomSolution.append(random.randint(0, 1))
            currentSolution = randomSolution
            currentSolutionFitness = calculateFitness(currentSolution)
            FITNESS_CALCULATIONS_COUNTER += 1
            print("Best solution fitness: (", FITNESS_CALCULATIONS_COUNTER, "/", MAX_FITNESS_CALCULATIONS, "):", currentSolution, currentSolutionFitness)


        if FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
            return currentSolution, currentSolutionFitness, FITNESS_CALCULATIONS_COUNTER


def VNS(initialSolution=None, initialFitness=None):
    global FITNESS_CALCULATIONS_COUNTER

    FITNESS_CALCULATIONS_COUNTER = 0

    if initialSolution:
        currentSolution = initialSolution
        currentSolutionFitness = initialFitness

    else:
        randomSolution = []


        for _ in range(0, len(X_train[0])):
            randomSolution.append(random.randint(0, 1))

        currentSolution = randomSolution
        currentSolutionFitness = calculateFitness(currentSolution)
        FITNESS_CALCULATIONS_COUNTER += 1
    bestSolution = currentSolution
    bestSolutionFitness = currentSolutionFitness


    nSize = 1

    while nSize <= 4:
        neighbours = []

        if nSize == 1:
            for index, value in enumerate(currentSolution):
                neighbour = currentSolution.copy()
                neighbour[index] = 1 - value
                neighbours.append(neighbour)

        elif nSize == 2:
            for c in range(len(currentSolution)):
                for r in range(c + 1, len(currentSolution)):
                    if c != r:
                        neighbour = currentSolution.copy()
                        neighbour[c], neighbour[r] = 1 - currentSolution[r], 1 - currentSolution[c]
                        neighbours.append(neighbour)


        elif nSize == 3:
            for c in range(len(currentSolution)):
                for r in range(c + 1, len(currentSolution)):
                    for m in range(r + 1, len(currentSolution)):
                        neighbour = currentSolution.copy()
                        neighbour[r], neighbour[c], neighbour[m] = 1 - currentSolution[c], 1 - currentSolution[m], 1 - currentSolution[r]
                        neighbours.append(neighbour)


        elif FITNESS_CALCULATIONS_COUNTER >= MAX_FITNESS_CALCULATIONS:
            break

        else:
            if nSize == 4:
                nSize = 1
                randomSolution = []
                for _ in range(len(X_train[0])):
                    randomSolution.append(random.randint(0, 1))
                currentSolution = randomSolution
                currentSolutionFitness = calculateFitness(currentSolution)
                FITNESS_CALCULATIONS_COUNTER += 1
                print("New best solution: (", FITNESS_CALCULATIONS_COUNTER, "/", MAX_FITNESS_CALCULATIONS, "):", currentSolution, currentSolutionFitness)


        bestNeighbour = currentSolution
        bestNeighbourFitness = currentSolutionFitness

        for neighbour in neighbours:
            neighbourFitness = calculateFitness(neighbour)
            FITNESS_CALCULATIONS_COUNTER += 1


            if neighbourFitness > bestNeighbourFitness:
                bestNeighbour = neighbour
                bestNeighbourFitness = neighbourFitness

        if bestNeighbourFitness > currentSolutionFitness:
            currentSolution = bestNeighbour
            currentSolutionFitness = bestNeighbourFitness
            nSize = 1

            if currentSolutionFitness > bestSolutionFitness:
                bestSolution = currentSolution
                bestSolutionFitness = currentSolutionFitness
                print("Best solution fitness: (",FITNESS_CALCULATIONS_COUNTER,"/",MAX_FITNESS_CALCULATIONS,")", bestSolution, bestSolutionFitness)

        nSize += 1

    return bestSolution, bestSolutionFitness, FITNESS_CALCULATIONS_COUNTER


#TODO: Write your algorithm as a funciton. You can add input parameters if you want.
def yourAlgorithm():
    global FITNESS_CALCULATIONS_COUNTER

    currentSolution, currentSolutionFitness, _ = hill_climbing()
    bestSolution, bestSolutionFitness, _ = VNS(initialSolution=currentSolution, initialFitness=currentSolutionFitness)

    return bestSolution, bestSolutionFitness


test()
test2()

#currentSolution, currentSolutionFitness, _ = hill_climbing()
#bestSolution, bestSolutionFitness, FITNESS_CALCULATIONS_COUNTER = VNS()
bestSolution, bestSolutionFitness = yourAlgorithm()
print("Best global solution: ", bestSolution, " Fitness: ", bestSolutionFitness)
