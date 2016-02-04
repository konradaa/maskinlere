# Authur: Konradaa, NTNU

#===============================================================
#   Oving 1 - TDT4173
#===============================================================
import numpy as np
import csv
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#==============================================================
#   Read data, last column is Y. 
#==============================================================
def read(filename):
    print ("start reading file...")
#   f = open(filename, 'r')
#   w = [float(x) for x in f.readline().strip(',')]
#   for line in f.readlines(): 
#       print (line.strip(','))

    with open(filename, 'r') as f: 
        reader = csv.reader(f, delimiter=',')
        w_ = ([[float(x.strip(',')) for x in row] for row in reader])
    
    x_ = np.asarray([[w_[x][0], w_[x][1]] for x in range(len(w_))]) #TODO: hardcode
    y_ = np.asarray([w_[x][-1] for x in range(len(w_))])
    return x_, y_


def linear_regression(X, Y): 
    print("start linear_regression...")
    random = np.random

#==============================
#    Test Data from Interwebs
#==============================
#   X = np.asarray([3,4,5,6.1,6.3,2.88,8.89,5.62,6.9,1.97,8.22,9.81,4.83,7.27,5.14,3.08])
#   Y = np.asarray([0.9,1.6,1.9,2.9,1.54,1.43,3.06,2.36,2.3,1.11,2.57,3.15,1.5,2.64,2.20,1.21])

    x = T.matrix('x')
    y = T.vector('y')
    b_value = 1.0#random.rand()
    w_value = 1.0#random.rand()

    b_ = theano.shared(b_value, name='b')
    w_ = theano.shared(w_value, name='w')
    n = X.shape[0] * 2

    #MSE
    pred = T.sum(T.dot(x,w_), axis=1) + b_      #equation 1
    los_function = T.sum(T.pow(pred - y,2))/n   #equation 2
    
#=============================
#   Test Function
#=============================
#    test_ =  theano.function([x], pred)

#=============================
#   Gradient
#=============================
    grad_w = T.grad(los_function*2,w_)
    grad_b = T.grad(los_function*2,b_)
   
#=============================
#   Setting rates
#=============================
    learning_rate       = 0.002
    training_rounds     = 800

#=============================
#   task a
#=============================
    print("Initial Parameters: ")
    print("w: " + str(w_value))
    print("b: " + str(b_value))
    print("a: " + str(learning_rate))
    
    
#=============================
#   Build function
#=============================
    train = theano.function([x,y], los_function, updates=[(w_,w_-learning_rate*grad_w),(b_,b_-learning_rate*grad_b)])
    test = theano.function([x], pred)

#=============================
#   Run training
#=============================
    f = []
    for i in range(training_rounds): 
        #TODO: plot los_function
        f.append(train(X,Y))
        if i == 4 or i ==9 or i == 0: 
            print("Iteration " + str(i+1) +": " + str(f))
#   print(f)
#   plt.plot(f)
#   plt.show()
    
#    print("W: " + str(w_.eval()))
#    print("b: " + str(b_.eval()))
    answ = test(X)
    
    print("Training ended")
#==================================
#   Start reading test data
#==================================
    filename = "data/data-test.csv"
    X,Y = read(filename)
    print("Running test")
    answ = test(X)
    print("Test ended")


#=================================
#   Start drawing Diagram
#================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],Y, c=["blue"])
    ax.scatter(X[:,0],X[:,1],answ, c = "red")
#   fig.plot(answ, projection='3d')
    fig.show()

    print(answ)
    import ipdb
    ipdb.set_trace()


def loss_function(w, a): 
    pass

filename = "data/data-test.csv"
X,Y = read(filename)
linear_regression(X,Y)
