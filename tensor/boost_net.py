import tensorflow as tf
import numpy as np
from net import Net
from Net.tensor import izzynet_boost, input_b


class boostNet():
	'''
	A dimensionality reduction layer followed by a weak learners
	'''

	def __init__(self, iters = 1):
        self.iters = iters
     

    def fit_reg(self, train_path, test_path, iters):
    	'''
        Trains a regression based boosting model 
        '''
        #Create Input Data Object
        data = input_b.IzzyData_B(train_path,test_path)

        ### TRAIN learner

        for i in range(self.iters):
            if(i == 0):
                net_w = izzynet_boost.IzzyNet_W()
                net.optimize(500,data, batch_size=200)
            else: 
                net_w = izzynet_boost.IzzyNet_W()
                #Need to load data
                net.optimize(500,data, batch_size=200)
    
    
            ### GET ERROR
            weights = get_error(data,net)

            ### UPDATE model


            ### UPDATE DISTRIBUTION
            data.update_weights(weights)




    def get_error(data,net):
        X,Y = data.train_data()
        weights = np.zeros(len(X))
        for i in len(X):
            weights[i] = LA.norm(net.predict(X[i])-Y(i))

        return weights