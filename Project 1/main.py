'''
Created on Sep 26, 2016

@author: maruthi
'''

#from pandas import DataFrame, read_csv
from xlrd import open_workbook
#import matplotlib.pyplot as plt
#import pandas as pd 
import sys 
import matplotlib 
import scipy.stats as st
import numpy as np

dataBook = open_workbook('university data.xlsx')
dataSheet = dataBook.sheet_by_index(0)

def getColValueByIndex(indexNumber):
    colValue = []
    for i in range(1,dataSheet.nrows):
        if dataSheet.cell_value(i,indexNumber)!= '':
            colValue.append(dataSheet.cell_value(i,indexNumber))
    return colValue

'''
Location = r'C:\Non OS Related Files\UB\Semester 1\Introduction to Machine Learning\Project Files\Project 1\university data.xlsx'
univ_dataFrame = pd.read_excel(Location)
'''
#csScoreArray = np.array(univ_dataFrame["CS Score (USNews)"])

import math
def getNaNIndices(arrayOfScores):
    count = 0
    indexArr = []
    for n in arrayOfScores:
        if math.isnan(n):
            indexArr.append(count)
        count = count + 1
    return indexArr

def getCalculatedMetrics(arrayOfScores, val):
    arrayOfScores = np.delete(arrayOfScores,getNaNIndices(arrayOfScores))
    arrayMean = np.mean(arrayOfScores)
    arrayVariance = np.var(arrayOfScores)
    arrayStd = np.std(arrayOfScores)
    metrics = { 'mean':arrayMean, 'variance':arrayVariance, 'standardDeviation':arrayStd, 'arrayOfScores' : arrayOfScores }
    return metrics

print "UBitName = maruthim"
print "personNumber = 50208681"

#variables
'''
csScore = np.array(univ_dataFrame["CS Score (USNews)"])
researchOverhead = np.array(univ_dataFrame["Research Overhead %"])
adminBasePayList = np.array(univ_dataFrame["Admin Base Pay$"])
tuitionMeanValuesList = np.array(univ_dataFrame["Tuition(out-state)$"])
'''

csScore = np.array(getColValueByIndex(2))
researchOverhead = np.array(getColValueByIndex(3))
adminBasePayList = np.array(getColValueByIndex(4))
tuitionMeanValuesList = np.array(getColValueByIndex(5))


#Mean calculation for the CS Score Array in the DataFrame from CSV File
metricCSScore = getCalculatedMetrics(csScore,1)
csScore = metricCSScore['arrayOfScores']
mu1 = np.around(metricCSScore["mean"],2)
var1 = np.around(metricCSScore["variance"],2)
sigma1 = np.around(metricCSScore["standardDeviation"],2)
#print 'mu1 ',metricCSScore["mean"]," var1 ",metricCSScore["variance"]," sigma1 ",metricCSScore["standardDeviation"]

#Mean calculation using the Research Overhead Array
metricResearchOverhead = getCalculatedMetrics(researchOverhead,2)
researchOverhead = metricResearchOverhead['arrayOfScores']
mu2 = np.around(metricResearchOverhead["mean"],2)
var2 = np.around(metricResearchOverhead["variance"],2)
sigma2 = np.around(metricResearchOverhead["standardDeviation"],2)
#print 'mu2 ',metricResearchOverhead["mean"]," var2 ",metricResearchOverhead["variance"]," sigma2 ",metricResearchOverhead["standardDeviation"]

#Mean calculation using the Admin Base Pay$ Array values
metricBasePay = getCalculatedMetrics(adminBasePayList,3)
adminBasePayList = metricBasePay['arrayOfScores']
mu3 = np.around(metricBasePay["mean"],2)
var3 = np.around(metricBasePay["variance"],2)
sigma3 = np.around(metricBasePay["standardDeviation"],2)
#print "mu3 ",metricBasePay["mean"]," var3 ",metricBasePay["variance"]," sigma3 ",metricBasePay["standardDeviation"]

#Mean calculation using the Tuition Array values
metricTuition = getCalculatedMetrics(tuitionMeanValuesList,4)
tuitionMeanValuesList = metricTuition['arrayOfScores']
mu4 = np.around(metricTuition["mean"],2)
var4 = np.around(metricTuition["variance"],2)
sigma4 = np.around(metricTuition["standardDeviation"],2)
#print "mu4 ",metricTuition["mean"]," var4 ",metricTuition["variance"]," sigma4 ", metricTuition["standardDeviation"]

print "mu1 = ",mu1
print "mu2 = ",mu2
print "mu3 = ",mu3
print "mu4 = ",mu4
print "var1 = ",var1
print "var2 = ",var2
print "var3 = ",var3
print "var4 = ",var4
print "sigma1 = ",sigma1
print "sigma2 = ",sigma2
print "sigma3 = ",sigma3
print "sigma4 = ",sigma4

#Covariance Matrix for the four variables
covarianceMat = np.around(np.cov(np.vstack((csScore,researchOverhead,adminBasePayList,tuitionMeanValuesList))),2)
print "covarianceMat = \n",covarianceMat

#Correlation Matrix for the four variables
correlationMat = np.around(np.corrcoef(np.vstack((csScore,researchOverhead,adminBasePayList,tuitionMeanValuesList))),2)
print "correlationMat = \n",correlationMat

def calculateLogLikelihood(pdfFunction):
    total = 1
    for val in np.nditer(pdfFunction):
        total = total * val
    loglkhd = np.log(total)
    return loglkhd

#Probability Distribution function for alll the variables csScore, researchOverhead, adminBasePayList, tutionMeanValuesList
pdfCsScore = st.norm.pdf(csScore, metricCSScore["mean"] , metricCSScore["standardDeviation"])
pdfResearchOverhead = st.norm.pdf(researchOverhead, metricResearchOverhead["mean"], metricResearchOverhead["standardDeviation"])
pdfAdminBasePayRate = st.norm.pdf(adminBasePayList, metricBasePay["mean"], metricBasePay["standardDeviation"])
pdfTutionMeanValue = st.norm.pdf(tuitionMeanValuesList, metricTuition["mean"], metricTuition["standardDeviation"])


#=========================Calculate the Loglikelihood======================
csScoreLogLkhd = calculateLogLikelihood(pdfCsScore)
researchOverheadLogLkhd = calculateLogLikelihood(pdfResearchOverhead)
adminBasePayListLogLkhd = calculateLogLikelihood(pdfAdminBasePayRate)
tuitionMeanValuesListLogLkhd = calculateLogLikelihood(pdfTutionMeanValue)
#calculating total log likelihood
logLikelihood = np.around((csScoreLogLkhd + researchOverheadLogLkhd + adminBasePayListLogLkhd + tuitionMeanValuesListLogLkhd),2)
print "logLikelihood = ",logLikelihood
likelihoodMaster = [csScoreLogLkhd, researchOverheadLogLkhd, adminBasePayListLogLkhd, tuitionMeanValuesListLogLkhd]


#-----------------------------BAYESIAN NETWORK CONSTRUCTION AND LOG LIKELIHOOD-----------------------------------------------------------
#Formulating Bayesian Network
#assuming Cs Score is dependent on other parameters
#Cs Score would be the variable Y dependent on X1, X2, X3 and an introduced variable X0

#===============Graph representation of the Bayesian Network=======================#
bayesianGraph = {
        'csScore' : [],
        'researchOverhead' : ['csScore'],
        'adminBasePay' : ['csScore'],
        'tuitionMeanValue' : ['csScore']
    }
bayesianVariables = ['csScore', 'researchOverhead', 'adminBasePay' , 'tuitionMeanValue']
bayesianMatrix = np.zeros((len(bayesianVariables), len(bayesianVariables)))
for v in range(len(bayesianVariables)):
    mappingForVariable = bayesianGraph[bayesianVariables[v]]
    for i in range(len(bayesianVariables)):
        if bayesianVariables[i] in mappingForVariable:
            bayesianMatrix[v][i] = 1
BNgraph = bayesianMatrix
print "BNgraph = \n",BNgraph


#Define a function to calculate the Bayesian Network Metrics
def calculateBayesianNetworkMetrics(yVar, xVar1, xVar2, xVar3, dependant):
    #Additional Data to be introduced as a X0 variable
    additionalData = np.zeros(yVar.size)
    for n in range(additionalData.size):
        additionalData[n] = 1.0
    masterData = [additionalData, xVar1, xVar2, xVar3]
    coeffMatrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            sumVal = 0
            for k in range(yVar.size):
                sumVal = sumVal + (masterData[i][k]*masterData[j][k])
                #print " sum ",sumVal
            coeffMatrix[i][j] = sumVal

    yVariableMatrix = np.zeros((4, 1))
    for n in range(4):
        sumVal = 0
        for i in range(yVar.size):
            sumVal = sumVal + (masterData[n][i] * yVar[i])
        yVariableMatrix[n] = sumVal

    #print "Coefficient Matrix ",coeffMatrix
    #print "Y Function Matrix ",yVariableMatrix

    betaArray = np.linalg.solve(coeffMatrix, yVariableMatrix);
    #print "beta weight founder Array \n",betaArray

    #Calculating the variance
    sumOfsquare = 0
    for n in range(yVar.size):
        sumLocal = 0
        for i in range(4):
            sumLocal = sumLocal + (masterData[i][n]*betaArray[i][0])
        sumLocal = sumLocal - yVar[n]
        sumOfsquare = sumOfsquare + (sumLocal * sumLocal)
    standardDeviation = np.sqrt((sumOfsquare/yVar.size))                  #Sigma Square is the Variance.
    varianceVal = (sumOfsquare/yVar.size)
    #print "standardDeviation ",standardDeviation
    #print "variance ",varianceVal
    
    #----------------------------------------------------------------------
    #Log Likelihood of Bayesion MODEL 1 using CS Score as dependent variable
    lglkhdConstant = -(0.5 * np.log(2*np.pi*varianceVal))
    #print "Constant ",lglkhdConstant
    BNlogLikelihood = 0;
    for n in range(yVar.size):
        sumVal = 0
        BNlogLikelihood = BNlogLikelihood + lglkhdConstant #Adding the constant part
        for i in range(masterData.__len__()):
            sumVal = sumVal + (masterData[i][n] * betaArray[i][0])    #Iterating over the sample list
        sumVal = sumVal - yVar[n]
        BNlogLikelihood = BNlogLikelihood + (-(sumVal*sumVal)/(2*varianceVal))
    #print "BN Log Likelihood ",BNlogLikelihood
    for i in range(likelihoodMaster.__len__()):
        if i!=dependant:
            BNlogLikelihood = BNlogLikelihood + likelihoodMaster[i]
    BNlogLikelihood = np.around(BNlogLikelihood,2)
    print "BNlogLikelihood = ",BNlogLikelihood       


#Calculate Bayesian Network Likelihood
calculateBayesianNetworkMetrics(csScore,researchOverhead,adminBasePayList,tuitionMeanValuesList, 0)
#calculateBayesianNetworkMetrics(researchOverhead, csScore, adminBasePayList, tuitionMeanValuesList, 1)
#calculateBayesianNetworkMetrics(adminBasePayList, csScore, researchOverhead, tuitionMeanValuesList, 2)
#calculateBayesianNetworkMetrics(tuitionMeanValuesList, csScore, researchOverhead, adminBasePayList, 3)

#================PLOT GENERATION FOR PAIRWISE DATA=================
'''
plt.figure(1)                #Initializing the figure object        

plt.subplot(321)             # 1st figure
plt.scatter(csScore,researchOverhead)
plt.xlabel("CS Score")
plt.ylabel("Research Overhead")

plt.subplot(322)             # 2nd figure
plt.scatter(researchOverhead,adminBasePayList)
plt.xlabel("Research Overhead")
plt.ylabel("Admin Base Pay")
            
plt.subplot(323)             # 3rd figure
plt.scatter(adminBasePayList,tuitionMeanValuesList)
plt.xlabel("Admin Base Pay")
plt.ylabel("Tuition(Out-State)")
  
plt.subplot(324)              
plt.scatter(csScore,adminBasePayList)             # 4th figure
plt.xlabel("CS Score")
plt.ylabel("Admin Base Pay")

plt.subplot(325)             # 5th figure
plt.scatter(csScore, tuitionMeanValuesList)
plt.xlabel("CS Score")
plt.ylabel("Tuition(Out-State)")

plt.subplot(326)             # 6th figure
plt.scatter(researchOverhead, tuitionMeanValuesList)
plt.xlabel("Research Overhead")
plt.ylabel("Tuition(Out-State)")

plt.show()  
'''