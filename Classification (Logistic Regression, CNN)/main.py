# -*- coding: utf-8 -*-
'''
Created on Nov 29, 2016

@author: marut
'''
import cPickle
import gzip
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#The Below file that is being read for the Neural Network training data we have downloaded it from the below link
# ---- https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/data  ---

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
'''
Setting up the training data into the input parameters and the output label parameters
'''
#---Training data---------
training_data_x = training_data[0]
training_data_t = training_data[1]

#---Testing data-----------
testing_data_x = test_data[0]
testing_data_t = test_data[1]

#---Validation data--------
validation_data_x = validation_data[0]
validation_data_t = validation_data[1]

#=================================================
#Fetching the USPS data
#=================================================
uspsData_x = []
uspsData_t = []
for i in xrange(0,10):
    for imageFile in glob.glob("USPSdata/Numerals/"+str(i)+"/*.png"):
        img = Image.open(imageFile)
        img = img.resize((28,28))
        uspsData_x.append(list(img.getdata()))
        uspsData_t.append(i)

sampleNumber = len(uspsData_x)
uspsData_norm_x = np.zeros((sampleNumber, 784))
uspsData_norm_t = np.zeros((sampleNumber, 1), dtype=np.int)

for i in xrange(sampleNumber):
    # Arranging the target data in an array of 19999 X 1
    uspsData_norm_t[i][0] = uspsData_t[i]
    for j in xrange(784):
        # Arranging the input vector data in an array of 19999 X 255
        uspsData_norm_x[i][j] = 1 - (float(uspsData_x[i][j])/float(255))



print "Name       : Maruthi Mohan Reddy Putha"
print "UBIT Number: 50208681"
#Calculating the Value of prediction along with weights
#Logistic Regression
num_of_Iterations = 15

def one_hot_encode_TARGET(targetArr):
    OHE_array = np.zeros((targetArr.shape[0], 10))
    for x in xrange(0, targetArr.shape[0]):
        OHE_array[x][targetArr[x]] = 1
    return OHE_array

def getExpSum(inputArr):
    sum_Arr = 0
    for x in xrange(0, inputArr.shape[0]):
        sum_Arr += np.exp(inputArr[x])
    return sum_Arr
    
def probability_Ck(predictedValues):        
    predictedProbability_Y = np.zeros((predictedValues.shape[0], predictedValues.shape[1]))        
    predictedExpSum = np.zeros((predictedProbability_Y.shape[0], predictedProbability_Y.shape[1]))
    for x in xrange(0,predictedProbability_Y.shape[0]):
        predictedExpSum[x] = np.sum(np.exp(predictedValues[x]))
        #print " shape of activation_k ", activation_k.shape
    for j in xrange(0, predictedProbability_Y.shape[0]):
        predictedProbability_Y[j] = np.exp(predictedValues[j])/predictedExpSum[j]
    return predictedProbability_Y

'''
This method is using the step wise processing of data instead of the block processing
'''
def getUpdatedWeight_RowWise(Iter_number):
    #Initializing the weights to 1 for all the cell values.
    weight_init = np.ones((training_data_x.shape[1],10))    
    learningRate = 0.01
    weight_updated = weight_init
    OHE_training_data_t = one_hot_encode_TARGET(training_data_t)
    #--------------------------------------------------------------------------------
    for x in xrange(0, Iter_number): 
        #print "Iteration Number ",x
        for y in xrange(0, training_data_x.shape[0]):
            #Forming the correct shapes of the matrices
            train_x = np.reshape(training_data_x[y], (1, training_data_x[y].shape[0]))
            train_t = np.reshape(OHE_training_data_t[y], (1, OHE_training_data_t[y].shape[0]))
            #Activation Calculation without the bias value added to it
            a_y = np.dot(train_x, weight_init)
            #Bias vector initialization
            b_y = np.ones((1, 10))
            #Updating the value of a_y
            a_y = np.add(a_y, b_y)
            #Calculation of probability of the K classes
            pred_y = np.exp(a_y)/float(np.sum(np.exp(a_y)))
            #Gradient of Error
            err_j = np.dot( np.transpose(train_x), np.subtract(pred_y, train_t) )
            #Updating the weight
            weight_updated = weight_init - ( learningRate * (err_j) )
            #Update the Weight_init
            weight_init = weight_updated
    #----- RETURN THE UPDATED WEIGHTS -----
    return (weight_init)

weightCalculated = getUpdatedWeight_RowWise(num_of_Iterations)

def evaluate_Model(weight_optimal, dataSet, dataLabelSet):
    #Activation without Bias value added to it
    activation_Y = np.dot(dataSet, weight_optimal)
    #Initializing the bias vector with all entries initialized to ones
    bias_Vector = np.ones((dataSet.shape[0], 10))
    #Activation at k, calculated a_k = X DOT W + Bias    
    activation_Biased_Y = np.add(activation_Y, bias_Vector)
    #Finding the probability for using the value
    prediction_Y = probability_Ck(activation_Biased_Y)    
    match_count = 0  
    predicted_target = []          
    for x in xrange(0, prediction_Y.shape[0]):
        predict_y = np.argmax(prediction_Y[x])
        predicted_target.append(predict_y)
        if predict_y == dataLabelSet[x]:
            match_count += 1
    correctness_Percentage = float(match_count)/float(prediction_Y.shape[0])*100
    #print "Correctness Percentage ",correctness_Percentage
    return (correctness_Percentage, prediction_Y, predicted_target)
uspsData_norm_t_OHE = one_hot_encode_TARGET(uspsData_norm_t)
print "Hyper Parameters"
print "Learning Rate: 0.01"
print "-----------------------------------------------"
print "Logistic Regression Evaluation on MNIST data :"
print "-----------------------------------------------"
evalMetrics_train = evaluate_Model(weightCalculated, training_data_x, training_data_t)
print "Correctness Percentage for training data : ",evalMetrics_train[0]
evalMetrics_valid = evaluate_Model(weightCalculated, validation_data_x, validation_data_t)
print "Correctness Percentage for validation data : ",evalMetrics_valid[0]
evalMetrics_test = evaluate_Model(weightCalculated, testing_data_x, testing_data_t)
print "Correctness Percentage for testing data : ",evalMetrics_test[0]
print "------------------------------------------------"
print "         Evaluation on USPS Data                "
print "------------------------------------------------"
evalMetrics_USPS = evaluate_Model(weightCalculated, uspsData_norm_x, uspsData_norm_t)
print "Correctness Percentage for testing data : ",evalMetrics_USPS[0]

#---------------------------------------------
#Neural Network: Single Layered Neural Network
#---------------------------------------------
iter_num_NN = 5
nodes_ConnectedLayer = 100
'''
This method is a new method that implements the derivative of the sigmoid while calculation of the delta of j
'''
def trainModel_Weights_Sigmoid(iter_num):
    # Initial Weights Setup and Bias Vector Setup along with the learning rate of 0.01
    #-------------------------------------------------------------------------------------------
    weight_init_ji = np.array(np.random.rand(training_data_x.shape[1], nodes_ConnectedLayer))*0.1
    weight_init_kj = np.array(np.random.rand(nodes_ConnectedLayer, 10))*0.1
    bias_vector_j = np.array(np.ones((training_data_x.shape[0], nodes_ConnectedLayer))) *0.1            # This is the BIAS Vector for i -> j which is a unit vector
    bias_vector_k = np.array(np.ones((training_data_x.shape[0], 10)))*0.1                               # This is the BIAS Vector for j -> k which is a unit vector
    training_x = np.array(training_data_x)
    
    learningRate_n = 0.01
    OHE_training_data_t = np.array(one_hot_encode_TARGET(training_data_t))
    #-------------------------------------------------------------------------------------------
    for x in xrange(0, iter_num): 
        #print 'Iteration number ',x    
        for j in xrange(0, training_data_x.shape[0]):           
            #Reshaping the necessary vectors to required shape
            train_x = np.reshape(training_x[j], (1, training_x[j].shape[0]))
            bias_j = np.reshape(bias_vector_j[j], (1, bias_vector_j[j].shape[0]))
            bias_k = np.reshape(bias_vector_k[j], (1, bias_vector_k[j].shape[0]))
            train_t = np.reshape(OHE_training_data_t[j], (1, OHE_training_data_t[j].shape[0]))
            #Calculation of Z_k
            z_j = np.add(np.dot(train_x, weight_init_ji), bias_j)
            sigmoid_z_j = 1/(1+np.exp(-z_j))
            #Activation
            a_k = np.add(np.dot( sigmoid_z_j, weight_init_kj ), bias_k)
            #Probability of Classes
            a_k_sum = np.sum(np.exp(a_k))
            y_k = np.exp(a_k)/(a_k_sum)             #Probability of the classes for K = 10          
            # Delta Calculation
            del_k = y_k - train_t            
            h_dash_z = (np.dot(1-sigmoid_z_j, np.transpose(sigmoid_z_j)))   #It is a scalar value = H'(aj) (As per the text book)
            del_j = h_dash_z[0][0]*(np.dot(del_k, np.transpose(weight_init_kj))) 
            err_j = np.dot(np.transpose(train_x), del_j)
            err_k = np.dot(np.transpose(sigmoid_z_j), del_k)
            
            weight_updated_ji = weight_init_ji - ((learningRate_n * err_j))
            weight_updated_kj = weight_init_kj - ((learningRate_n * err_k))
                       
            weight_init_ji = weight_updated_ji
            weight_init_kj = weight_updated_kj
            
    #--------- RETURN THE WEIGHTS UPDATED ---------------
    return (weight_init_ji, weight_init_kj)       
        
#weightsCalcualted_NN = trainModel_Weights(num_of_Iterations)  -- A Method without including the Sigmoid Derivative
weightsCalcualted_NN = trainModel_Weights_Sigmoid(iter_num_NN)
    
def evaluate_NN_Model(weightSet, dataSet, dataLabelSet):
    bias_vector_j = np.ones((dataSet.shape[0], nodes_ConnectedLayer))             # This is the BIAS Vector for i -> j which is a unit vector
    bias_vector_k = np.ones((dataSet.shape[0], 10))                               # This is the BIAS Vector for j -> k which is a unit vector
    #Calculation of Z_k
    firstLayer_z = np.add(np.dot(dataSet, weightSet[0]), bias_vector_j)
    sigmoidOf_z = 1/(1+np.exp(-firstLayer_z))
    #Calculation of Y_k
    activation_k = np.add(np.dot(sigmoidOf_z, weightSet[1]), bias_vector_k)
    y_predicted = probability_Ck(activation_k)
    match_count = 0  
    predicted_target = []          
    for x in xrange(0, dataLabelSet.shape[0]):
        predict_y = np.argmax(y_predicted[x])
        predicted_target.append(predict_y)
        if predict_y == dataLabelSet[x]:
            match_count += 1
    correctness_Percentage = float(match_count)/float(dataLabelSet.shape[0])*100
    #print "Correctness Percentage ",correctness_Percentage
    return (correctness_Percentage, y_predicted, predicted_target)

#evalMetrics_NN = evaluate_NN_Model(weightsCalcualted_NN, training_data_x, training_data_t)
print "Hyper Parameters"
print "Learning Rate: 0.01"
print "Number of nodes in the Hidden Layer: 100"
print "----------------------------------------------------------"
print "Single Layered Neural Networks Evaluation on MNIST data :"
print "----------------------------------------------------------"
evalMetrics_NN_train = evaluate_NN_Model(weightsCalcualted_NN, training_data_x, training_data_t)
print "Correctness Percentage for training data : ",evalMetrics_NN_train[0]
evalMetrics_NN_valid = evaluate_NN_Model(weightsCalcualted_NN, validation_data_x, validation_data_t)
print "Correctness Percentage for validation data : ",evalMetrics_NN_valid[0]
evalMetrics_NN_test = evaluate_NN_Model(weightsCalcualted_NN, testing_data_x, testing_data_t)
print "Correctness Percentage for testing data : ",evalMetrics_NN_test[0]
print "------------------------------------------------"
print "         Evaluation on USPS Data                "
print "------------------------------------------------"
evalMetrics_NN_USPS = evaluate_NN_Model(weightsCalcualted_NN, uspsData_norm_x, uspsData_norm_t)
print "Correctness Percentage for testing data : ",evalMetrics_NN_USPS[0]

#------------------------------------------------------------------
#   Convolutional Neural Networks
#------------------------------------------------------------------
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(2000):
	batch = mnist.train.next_batch(50)
	if i%500 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		#print "step %d, training accuracy %g"%(i, train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "-----------------------------------------------------------------------"
print "Convolutional Neural Networks"
print "-----------------------------------------------------------------------"
print "Testing Accuracy for MNIST Data is %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

print "Testing Accuracy for USPS Data is %g"%accuracy.eval(feed_dict={x: uspsData_norm_x, y_: uspsData_norm_t_OHE, keep_prob: 1.0})



