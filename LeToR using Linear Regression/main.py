'''
Created on Oct 21, 2016

@author: marut
'''

import numpy as np
import random as rd
import csv as csvHelper
#from dask.array.core import square
'''
def getArrayOfImportantCharacters(fileLine):
    splittedLine = fileLine.split(" ")
    isFirst = True
    for eachStr in (splittedLine[0:48]):
        if isFirst:
            print eachStr
            isFirst = False
        else:
            labelAndValue = eachStr.split(":")
            print labelAndValue[0]+" : "+labelAndValue[1]
'''            
''' ---------------------------- READING THE SYNTHETIC DATA BEGINS ----------------------------------------------------------'''
yData_Synth = "output.csv"
xData_Synth = "input.csv"

synth_NumOfFeatures = 10
synth_trainingDataCount = 16000
synth_validationDataEnd = 18000
synth_testingDataEnd = 20000

# SETTING UP THE MATRICES FOR STORING THE FEATURE VALUES AND RELEVANCE LABELS
synth_relevanceLabelMatrix_TRAIN = np.zeros((synth_trainingDataCount,1))
synth_featureMatrix_TRAIN = np.zeros((synth_trainingDataCount,synth_NumOfFeatures))
#----------------------------------------------------------------------
synth_relevanceLabelMatrix_VALID = np.zeros((synth_validationDataEnd-synth_trainingDataCount,1))
synth_featureMatrix_VALID = np.zeros((synth_validationDataEnd-synth_trainingDataCount,synth_NumOfFeatures))
#----------------------------------------------------------------------
synth_relevanceLabelMatrix_TEST = np.zeros((synth_testingDataEnd-synth_validationDataEnd,1))
synth_featureMatrix_TEST = np.zeros((synth_testingDataEnd-synth_validationDataEnd,synth_NumOfFeatures))

synth_relevanceLabelMatrix_Master = []
synth_featureMatrix_Master = []

''' WRITING THE RELEVANCE LABELS TO THE RELEVANT ARRAYS '''
with open(yData_Synth, 'rU') as csvfile:
    fileReader = csvHelper.reader(csvfile, dialect=csvHelper.excel_tab)
    for y_Master in fileReader:
        synth_relevanceLabelMatrix_Master.append(y_Master)
with open(xData_Synth, 'rU') as csvfile:
    fileReader = csvHelper.reader(csvfile, dialect=csvHelper.excel_tab)
    for x_Master in fileReader:
        synth_featureMatrix_Master.append(x_Master)

# WRITING THE RELEVANCE LABELS TO THE TRAINING SET
initialIndex = 0    #resetting the index to zero after every iteration
for y in synth_relevanceLabelMatrix_Master[0:synth_trainingDataCount]:
    #print y
    synth_relevanceLabelMatrix_TRAIN[initialIndex] = float(y[0])
    initialIndex = initialIndex + 1
initialIndex = 0    #resetting the index to zero after every iteration
for y in synth_relevanceLabelMatrix_Master[synth_trainingDataCount:synth_validationDataEnd]:
    synth_relevanceLabelMatrix_VALID[initialIndex] = float(y[0])
    initialIndex = initialIndex + 1
initialIndex = 0    #resetting the index to zero after every iteration
for y in synth_relevanceLabelMatrix_Master[synth_validationDataEnd:synth_testingDataEnd]:
    synth_relevanceLabelMatrix_TEST[initialIndex] = float(y[0])
    initialIndex = initialIndex + 1  
  
# WRITING THE FEATURES OF THE TRAINING SET
initialIndex = 0    #resetting the index to zero after every iteration
for x in synth_featureMatrix_Master[0:synth_trainingDataCount]:
    x = x[0]
    singleRow = x.split(",")
    for f in xrange(0, synth_NumOfFeatures):
        synth_featureMatrix_TRAIN[initialIndex][f] = float(singleRow[f])
    initialIndex = initialIndex + 1 
# WRITING THE FEATURES OF THE VALIDATION SET
initialIndex = 0    #resetting the index to zero after every iteration
for x in synth_featureMatrix_Master[synth_trainingDataCount:synth_validationDataEnd]:
    x = x[0]
    singleRow = x.split(",")
    for f in xrange(0, synth_NumOfFeatures):
        synth_featureMatrix_VALID[initialIndex][f] = float(singleRow[f])
    initialIndex = initialIndex + 1 
# WRITING THE FEATURES OF THE TESTING SET
initialIndex = 0    #resetting the index to zero after every iteration
for x in synth_featureMatrix_Master[synth_validationDataEnd:synth_testingDataEnd]:
    x = x[0]
    singleRow = x.split(",")
    for f in xrange(0, synth_NumOfFeatures):
        synth_featureMatrix_TEST[initialIndex][f] = float(singleRow[f])
    initialIndex = initialIndex + 1 
    
#print synth_featureMatrix_TRAIN.shape
#print synth_relevanceLabelMatrix_TRAIN.shape

''' ------------------------------------------ END OF READING SYNTHETIC DATA --------------------------------------'''    

''' ----------------------------------------- READING THE MICROSOFT DATA ------------------------------------------''' 

msDataFilePath = "Querylevelnorm.txt"
# SETTING UP THE BOUNDARIES FOR TRAINING, VALIDATION AND TESTING DATA FROM MASTER DATA
numOfFeatures = 46
trainingDataCount = 55704
validationDataEnd = 62813 
testingDataEnd = 69923 

# SETTING UP THE MATRICES FOR STORING THE FEATURE VALUES AND RELEVANCE LABELS
relevanceLabelMatrix_TRAIN = np.zeros((trainingDataCount,1))
featureMatrix_TRAIN = np.zeros((trainingDataCount,numOfFeatures))
#----------------------------------------------------------------------
relevanceLabelMatrix_VALID = np.zeros((validationDataEnd-trainingDataCount,1))
featureMatrix_VALID = np.zeros((validationDataEnd-trainingDataCount,numOfFeatures))
#----------------------------------------------------------------------
relevanceLabelMatrix_TEST = np.zeros((testingDataEnd-validationDataEnd,1))
featureMatrix_TEST = np.zeros((testingDataEnd-validationDataEnd,numOfFeatures))

'''
    Sets up the test data and split it into Training Validation Testing sets
'''
def setUpData(wordArray, indexOfWord, isTrainingData, isValidationData, isTestingData):
    precisionLabel = wordArray[0]
    '''
    Adding the RelevanceLabel to the relevance label matrix
    '''
    if isTrainingData:
        relevanceLabelMatrix_TRAIN[indexOfWord] = float(precisionLabel)
    elif isValidationData:
        relevanceLabelMatrix_VALID[indexOfWord] = float(precisionLabel)
    elif isTestingData:
        relevanceLabelMatrix_TEST[indexOfWord] = float(precisionLabel)
        
    '''
    Getting the features from the word array
    '''
    wordArray = wordArray[2:48]
    for ftr in xrange(0,45):
        #Adding the features to the feature matrix
        if isTrainingData:
            featureMatrix_TRAIN[indexOfWord][ftr] = float(wordArray[ftr].split(":")[1])
        elif isValidationData:
            featureMatrix_VALID[indexOfWord][ftr] = float(wordArray[ftr].split(":")[1])
        elif isTestingData:
            featureMatrix_TEST[indexOfWord][ftr] = float(wordArray[ftr].split(":")[1])
      
with open(msDataFilePath, 'r') as f:
    data = f.readlines()
    indexCurrent = 0
    for line in data[0:trainingDataCount]:
        words = line.split()[0:48]        
        setUpData(words, indexCurrent, True, False, False)
        indexCurrent = indexCurrent + 1
        #print words
        
    indexCurrent = 0
    for line in data[trainingDataCount:validationDataEnd]:
        words = line.split()[0:48]
        setUpData(words, indexCurrent, False, True, False)
        indexCurrent = indexCurrent + 1
    
    indexCurrent = 0
    for line in data[validationDataEnd:testingDataEnd]:
        words = line.split()[0:48]
        setUpData(words, indexCurrent, False, False, True)
        indexCurrent = indexCurrent + 1

''' ---------------------------------------- END OF READING THE MICROSOFT DATA ----------------------------------------------'''

#numberOfBasisFns = int(raw_input("Choose the number of Basis Functions : "))
numberOfBasisFns = 6

''' A FUNCTION TO CALCULATE THE WEIGHTS USING THE RADIAL BASIS FUNCTIONS '''
def calculateWeights_RBF(dataSet, relevanceLabelSet, noOfFeatures, isRegularized):
    #numberOfPhi = input("Enter M value")
    numberOfPhi = numberOfBasisFns
    dataSetLength = dataSet.shape[0]

    meanArray = np.zeros((numberOfPhi-1, noOfFeatures))
    phiMatrix = np.zeros((dataSetLength, numberOfPhi-1))

    varianceMatrix = np.zeros((noOfFeatures, noOfFeatures))
    # FORMULATE THE VARIANCE MATRIX WITH VARIANCE IN THE DIAGONAL
    for num in xrange(0, noOfFeatures):
        tempColumn = []
        for num2 in xrange(0, noOfFeatures):
            tempColumn.append(dataSet[num2][num])
        currentVariance = np.var(tempColumn)
        if currentVariance != 0.0:
            varianceMatrix[num][num] = np.var(tempColumn)
        else:
            varianceMatrix[num][num] = 0.0001
        
    # FORMULATE THE MEAN VALUES IN THE DATA
    for num in xrange(0, numberOfPhi-1):
        #GENERATE MEANS RANDOMLY IN BETWEEN THE ENTIRE DATA SET
        meanArray[num] = dataSet[rd.randint(0, dataSetLength-1)]
        #print meanArray[num]
        #meanArray[num] = (dataSet[num])

    for i in xrange(0, dataSetLength):
        for j in xrange(0, numberOfPhi-1):
            #phiMatrix[i][j] = np.exp(1/2)      
            #print "Mean value ",meanArray[j]  
            subMatrix = np.subtract(dataSet[i], meanArray[j])            
            phiMatrix[i][j] = np.exp(((-0.5)*(np.dot(np.dot(np.transpose(subMatrix), np.linalg.inv(varianceMatrix)),subMatrix))))

    phiZero = np.ones((dataSetLength, 1),dtype=float)
    phiMatrix = np.hstack((phiZero, phiMatrix))

    #print "Shape of Phi Matrix ",phiMatrix.shape
    #Regularization Term
    '''
    lambdaI = 0.4 * np.identity(numberOfPhi) #Here the regularization term has been assumed to be 0.4
    
    phiTran = np.transpose(phiMatrix)
    
    weightMatrix = np.zeros((numberOfPhi, 1))
    
    if isRegularized is True:        
        weightMatrix = np.dot(np.dot(np.linalg.inv(np.add(lambdaI ,np.dot(phiTran,phiMatrix))),phiTran),relevanceLabelSet)
    else:
        weightMatrix = np.dot(np.dot(np.linalg.inv(np.dot(phiTran,phiMatrix)),phiTran),relevanceLabelSet)

    
    # CALCULATION OF rms ERROR FOR GAUSSIAN BASIS FUNCTIONS
    squareErrSum = 0
    for i in xrange(0, dataSetLength):
        squareErrSum = squareErrSum + np.square(relevanceLabelSet[i] - np.dot(np.transpose(weightMatrix), phiMatrix[i]))

    errRMS = np.sqrt(squareErrSum/dataSetLength)
    print "RMS Error for training data"
    print "Error Value ",squareErrSum
    print "RMS Error",errRMS
    '''
    return (phiMatrix)

''' A function used to calculate the weights for the radial basis functions '''
def calculate_Weights(phiMatrixCalculated, relevanceSetGiven, isRegularized):
    totalNumOfPhi = numberOfBasisFns
    # Regularization Term
    lambdaI = 0.4 * np.identity(totalNumOfPhi) #Here the regularization term has been assumed to be 0.4
    
    phiTran = np.transpose(phiMatrixCalculated)
    
    weightMatrix = np.zeros((totalNumOfPhi, 1))
    #print phiMatrixCalculated.shape, relevanceSetGiven.shape, weightMatrix.shape
    if isRegularized is True:        
        weightMatrix = np.dot(np.dot(np.linalg.inv(np.add(lambdaI ,np.dot(phiTran,phiMatrixCalculated))),phiTran),relevanceSetGiven)
    else:
        weightMatrix = np.dot(np.dot(np.linalg.inv(np.dot(phiTran,phiMatrixCalculated)),phiTran),relevanceSetGiven)    
    return weightMatrix

''' A FUNCTION USED TO CALCULATE THE RMS ERROR USING WEIGHTS FOUUND AND PHI MATRIX '''
def calculateRMSError(relevanceLabelSet, basisFunctionMatrix, weightMatrix):
    squareErrSum = 0
    lenOfDataSet = relevanceLabelSet.shape[0]
    for i in xrange(0, lenOfDataSet):
        squareErrSum = squareErrSum + np.square(relevanceLabelSet[i] - np.dot(np.transpose(weightMatrix), basisFunctionMatrix[i]))
    errRMS = np.sqrt(squareErrSum/lenOfDataSet)
    return errRMS

iterationStore = []
learningRateChange = []

def calculateWeights_GradientDescent(phiRBF, iterNo, relevanceLabelSet, isRegularized):    
    weightInput = np.ones((1,numberOfBasisFns))
    learningRate = 1 #Assume that the weight is growing at a slower rate
    weightCalculated = np.zeros((numberOfBasisFns,1))
    initialErms = 0
    
    for i in xrange(1, iterNo):
        # REGULARIZATION TERM
        lambdaRegularizer = 0.4 * weightInput
        errorStep = np.zeros((1, numberOfBasisFns))
        
        if isRegularized is True:
            errorStep = np.add(np.dot(np.transpose(np.subtract(relevanceLabelSet, np.dot(phiRBF, np.transpose(weightInput)))),phiRBF), lambdaRegularizer)
        else:
            errorStep = np.dot(np.transpose(np.subtract(relevanceLabelSet, np.dot(phiRBF, np.transpose(weightInput)))),phiRBF)
        
        errorGradient = -1* learningRate * (errorStep/trainingDataCount)    #Getting the negative gradient descent
        weightCalculated = np.subtract(weightInput, errorGradient)
        #weightCalculated.shape = (weightCalculated.shape[0],1)  #Setting the weight calculated to a different shape
        if initialErms==0:
            initialErms = calculateRMSError(relevanceLabelSet, phiRBF, weightCalculated[0])
        else:
            currentErms = calculateRMSError(relevanceLabelSet, phiRBF, weightCalculated[0])
            if currentErms > initialErms:   #Setting the learning to a lesser value if the Error grows further
                learningRate = (learningRate * 0.5) #If the learning is same as before or less than the earlier value keep it as it is
            errorDifference = (currentErms - initialErms)   #Store the Error difference in the local values
            if (errorDifference < 0):
                errorDifference = errorDifference * -1                
            if errorDifference <= 0.001:    #If the error difference is of the order of 0.001
                    break   #If the error difference is so small break the loop
        
        
        #Assign the weightCalculated to the weightInput
        weightInput = weightCalculated

    return (weightCalculated)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- CALCULATION FOR THE SYNTHETIC DATA SET STARTS HERE -------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print "Number of Basis Functions : 6"
print "Regularization Parameter : 0.4"
print "Learning Rate : 1 ( Starting Value )"

print "================================= CALCULATIONS DONE WITH REGULARIZATIOIN ==================================="
print "----------------------"
print "SYNTHETIC DATA"
print "----------------------"
# CALCULATE THE WEIGHTS USING THE RBF FOR SYNTHETIC DATA
rbf_Phi_One = calculateWeights_RBF(synth_featureMatrix_TRAIN, synth_relevanceLabelMatrix_TRAIN, synth_NumOfFeatures, True)
synth_phi_Valid = calculateWeights_RBF(synth_featureMatrix_VALID, synth_relevanceLabelMatrix_VALID, synth_NumOfFeatures, True)
synth_phi_Test = calculateWeights_RBF(synth_featureMatrix_TEST, synth_relevanceLabelMatrix_TEST, synth_NumOfFeatures, True)

# CALCULATE THE WEIGHTS FROM GAUSSIAN RADIAL DISTRIBUTION
print "---------------Weights calculated from Radial Gaussian Distribution ----------------"
synth_weight = calculate_Weights(rbf_Phi_One, synth_relevanceLabelMatrix_TRAIN, True)
print synth_weight

# CALCULATE RMS ERROR BASED ON THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(synth_relevanceLabelMatrix_TRAIN, rbf_Phi_One, synth_weight)
print "RMS_Validation ",calculateRMSError(synth_relevanceLabelMatrix_VALID, synth_phi_Valid, synth_weight)
print "RMS_Test       ",calculateRMSError(synth_relevanceLabelMatrix_TEST, synth_phi_Test, synth_weight)

# Calculate the values from the stochastic gradient method
synth_sgd_Weights = calculateWeights_GradientDescent(rbf_Phi_One, 1000, synth_relevanceLabelMatrix_TRAIN, True )
print "-------------Weights calculated from Stochastic Gradient Descent Method-------------------"
synth_sgd_wt = np.array(synth_sgd_Weights[0])
synth_sgd_wt.shape = (synth_sgd_wt.shape[0],1)
print synth_sgd_wt

# CALCULATE RMS ERROR BASED ON THE stochastic gradient method FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(synth_relevanceLabelMatrix_TRAIN, rbf_Phi_One, synth_sgd_Weights[0])
print "RMS_Validation ",calculateRMSError(synth_relevanceLabelMatrix_VALID, synth_phi_Valid, synth_sgd_Weights[0])
print "RMS_Test       ",calculateRMSError(synth_relevanceLabelMatrix_TEST, synth_phi_Test, synth_sgd_Weights[0])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------- CALCULATION FOR THE MICROSOFT DATA SET STARTS HERE -------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------"
print "LeToR DATASET"
print "-----------------------"
# CALCULATE THE WEIGHTS USING THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
rbf_Phi = calculateWeights_RBF(featureMatrix_TRAIN, relevanceLabelMatrix_TRAIN, numOfFeatures, True)
ms_phi_Valid = calculateWeights_RBF(featureMatrix_VALID, relevanceLabelMatrix_VALID, numOfFeatures, True)
ms_phi_Test = calculateWeights_RBF(featureMatrix_TEST, relevanceLabelMatrix_TEST, numOfFeatures, True)

dataSet_weight = calculate_Weights(rbf_Phi, relevanceLabelMatrix_TRAIN, True)
# CALCULATE THE WEIGHTS FROM GAUSSIAN RADIAL DISTRIBUTION
print "---------------Weights calculated from Radial Gaussian Distribution----------------"
print dataSet_weight

# CALCULATE RMS ERROR BASED ON THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(relevanceLabelMatrix_TRAIN, rbf_Phi, dataSet_weight)
print "RMS_Validation ",calculateRMSError(relevanceLabelMatrix_VALID, ms_phi_Valid, dataSet_weight)
print "RMS_Test       ",calculateRMSError(relevanceLabelMatrix_TEST, ms_phi_Test, dataSet_weight)

# CACLCULATE THE WEIGHTS USING THE STOCHASTIC GRADIENT DESCENT
sgd_Weights = calculateWeights_GradientDescent(rbf_Phi, 1000, relevanceLabelMatrix_TRAIN, True)
print " ---------------- Weight Calculated from Stochastic Gradient Descent ---------------------- "
gd_wt = np.array(sgd_Weights[0])
gd_wt.shape = (gd_wt.shape[0],1)
print gd_wt

print "RMS_Training   ",calculateRMSError(relevanceLabelMatrix_TRAIN, rbf_Phi, sgd_Weights[0])
print "RMS_Validation ",calculateRMSError(relevanceLabelMatrix_VALID, ms_phi_Valid, sgd_Weights[0])
print "RMS_Test       ",calculateRMSError(relevanceLabelMatrix_TEST, ms_phi_Test, sgd_Weights[0])

print "================================= CALCULATIONS DONE WITHOUT REGULARIZATIOIN ==================================="

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- CALCULATION FOR THE SYNTHETIC DATA SET STARTS HERE -------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print "----------------SYNTHETIC DATA-----------------------------"
# CALCULATE THE WEIGHTS USING THE RBF FOR SYNTHETIC DATA
rbf_Phi_One = calculateWeights_RBF(synth_featureMatrix_TRAIN, synth_relevanceLabelMatrix_TRAIN, synth_NumOfFeatures, False)
synth_phi_Valid = calculateWeights_RBF(synth_featureMatrix_VALID, synth_relevanceLabelMatrix_VALID, synth_NumOfFeatures, False)
synth_phi_Test = calculateWeights_RBF(synth_featureMatrix_TEST, synth_relevanceLabelMatrix_TEST, synth_NumOfFeatures, False)

# CALCULATE THE WEIGHTS FROM GAUSSIAN RADIAL DISTRIBUTION
print "---------------Weights calculated from Radial Gaussian Distribution ----------------"
synth_weight = calculate_Weights(rbf_Phi_One, synth_relevanceLabelMatrix_TRAIN, False)
print synth_weight

# CALCULATE RMS ERROR BASED ON THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(synth_relevanceLabelMatrix_TRAIN, rbf_Phi_One, synth_weight)
print "RMS_Validation ",calculateRMSError(synth_relevanceLabelMatrix_VALID, synth_phi_Valid, synth_weight)
print "RMS_Test       ",calculateRMSError(synth_relevanceLabelMatrix_TEST, synth_phi_Test, synth_weight)

# Calculate the values from the stochastic gradient method
synth_sgd_Weights = calculateWeights_GradientDescent(rbf_Phi_One, 1000, synth_relevanceLabelMatrix_TRAIN, False )
print "-------------Weights calculated from Stochastic Gradient Descent Method-------------------"
synth_sgd_wt = np.array(synth_sgd_Weights[0])
synth_sgd_wt.shape = (synth_sgd_wt.shape[0],1)
print synth_sgd_wt

# CALCULATE RMS ERROR BASED ON THE stochastic gradient method FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(synth_relevanceLabelMatrix_TRAIN, rbf_Phi_One, synth_sgd_Weights[0])
print "RMS_Validation ",calculateRMSError(synth_relevanceLabelMatrix_VALID, synth_phi_Valid, synth_sgd_Weights[0])
print "RMS_Test       ",calculateRMSError(synth_relevanceLabelMatrix_TEST, synth_phi_Test, synth_sgd_Weights[0])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------- CALCULATION FOR THE MICROSOFT DATA SET STARTS HERE -------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print"-----------------------LeToR DATASET------------------------"
# CALCULATE THE WEIGHTS USING THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
rbf_Phi = calculateWeights_RBF(featureMatrix_TRAIN, relevanceLabelMatrix_TRAIN, numOfFeatures, False)
ms_phi_Valid = calculateWeights_RBF(featureMatrix_VALID, relevanceLabelMatrix_VALID, numOfFeatures, False)
ms_phi_Test = calculateWeights_RBF(featureMatrix_TEST, relevanceLabelMatrix_TEST, numOfFeatures, False)

dataSet_weight = calculate_Weights(rbf_Phi, relevanceLabelMatrix_TRAIN, False)
# CALCULATE THE WEIGHTS FROM GAUSSIAN RADIAL DISTRIBUTION
print "---------------Weights calculated from Radial Gaussian Distribution----------------"
print dataSet_weight

# CALCULATE RMS ERROR BASED ON THE RADIAL GAUSSIAN FUNCTION FOR VARIOUS TEST DATA
print "RMS_Training   ",calculateRMSError(relevanceLabelMatrix_TRAIN, rbf_Phi, dataSet_weight)
print "RMS_Validation ",calculateRMSError(relevanceLabelMatrix_VALID, ms_phi_Valid, dataSet_weight)
print "RMS_Test       ",calculateRMSError(relevanceLabelMatrix_TEST, ms_phi_Test, dataSet_weight)

# CACLCULATE THE WEIGHTS USING THE STOCHASTIC GRADIENT DESCENT
sgd_Weights = calculateWeights_GradientDescent(rbf_Phi, 1000, relevanceLabelMatrix_TRAIN, False)
print " ---------------- Weight Calculated from Stochastic Gradient Descent ---------------------- "
gd_wt = np.array(sgd_Weights[0])
gd_wt.shape = (gd_wt.shape[0],1)
print gd_wt

print "RMS_Training   ",calculateRMSError(relevanceLabelMatrix_TRAIN, rbf_Phi, sgd_Weights[0])
print "RMS_Validation ",calculateRMSError(relevanceLabelMatrix_VALID, ms_phi_Valid, sgd_Weights[0])
print "RMS_Test       ",calculateRMSError(relevanceLabelMatrix_TEST, ms_phi_Test, sgd_Weights[0])


