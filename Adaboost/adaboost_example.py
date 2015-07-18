import numpy as np
import math

# Builds a decision tree stump. Not super usable for other data
def buildStump(data, labels, weights):
    signs = ["<", ">"]
    
    bestError = len(data)
    bestSign = None
    bestSplit = None

    # go through the data, -.5, .5, 1.5, ...
    for split in np.arange(-.5, len(data) + .5, 1):
        for sign in signs:
            #calculate the weighted error of this stump
            error = getClassifierError(data, labels, weights, (0, split, sign));
            if error < bestError:
                bestError = error
                bestSign = sign
                bestSplit = split
    return bestError, bestSplit, bestSign

# Given the data, labels, weights, and decision stump, tells you the weighted
# error for the classifier
def getClassifierError(data, labels, weights, stump):
    error = 0
    for i in xrange(len(data)):
        if classify(data[i], stump) != labels[i]:
            error += weights[i]
    return error

# Given a stump and data value, classifies it as +1 or -1
def classify(dataValue, stump):
    split = stump[1]
    sign = stump[2]
    if (sign == "<"):
        if (dataValue < split):
            return 1;
        else:
            return -1;
    else:
        if (dataValue > split):
            return 1;
        else:
            return -1;

# Runs the adaboost algorithm on the dataset a specified number of times
def boost(trainingIterations): 
    # lets get our data ready
    trainingData = np.array([0,1,2,3,4,5,6,7,8,9])
    trainingLabels = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
    weights = np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
    classifierWeights = []
    classifiers = []
    
    for t in range(trainingIterations):
        stump = buildStump(trainingData, trainingLabels, weights)
        classifiers.append(stump) # remember this classfier

        error = stump[0]
        alpha = .5 * math.log((1 - error) / error)
        classifierWeights.append(alpha) # keep track of this classifier's weight
        print "Error on round " + str(t) + " is: " + str(getClassifierError(trainingData, trainingLabels, weights, stump))
        
        for i in xrange(len(weights)):
            weights[i] *= math.exp(-alpha * trainingLabels[i] * classify(trainingData[i], stump))
        Z = sum(weights) # get the sum of the weights
        weights /= Z # divide by that to make all the weights add to one

    print "\nDone boosting!\n"

    # print our function
    function = "f_" + str(trainingIterations) + " = "
    for i in range(len(classifiers)):
        function += "{0:.5f}".format(classifierWeights[i]) + " * I(x " + classifiers[i][2] + " " + str(classifiers[i][1]) + ")"
        if (i < len(classifiers) - 1):
            function += " + "
    print function

    # calculate the total error
    totalErrors = 0
    for i in range(len(trainingData)):
        classifierDecision = 0
        for k in range(len(classifiers)):
            classifierDecision += classifierWeights[k] * classify(trainingData[i], classifiers[k])
        classifierDecision = np.sign(classifierDecision)

        if (classifierDecision != trainingLabels[i]):
            totalErrors += 1

    print "Total errors: " + str(totalErrors)
        
