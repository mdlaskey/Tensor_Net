import random
import numpy as np
import numpy.linalg as LA
import IPython
import tensorflow as tf
import cv2
import IPython
import sys
sys.path.append("../")

from plot_class import Plotter

class InputData():

    def __init__(self, train_d, test_d):
        

        random.shuffle(self.train_data)
        random.shuffle(self.test_data)


    def next_train_batch(self, n):
        if self.i + n > len(self.train_data):
            self.i = 0
            random.shuffle(self.train_data)
        batch = self.train_data[self.i:n+self.i]
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])
    
    def next_test_batch(self):
        batch = self.test_data
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])




def process_out(n):
    out = np.argmax(n)

    return out


class GridData_UB(InputData):
    def __init__(self, data,T,w=15,h=15):
        self.train_tups = []
        self.test_tups = []
        self.w = 15
        self.h = 15 
        self.i = 0
        self.T = T
        data_c = []
        self.states = []
        self.dist = []
        self.plotter = Plotter()
        self.weights = []
        self.densities = []

        t = 0
        old_traj = []
        #Create a list with each point being the whole trajectory
        for i in range(len(data)):
            traj = []
            traj.append([data[i][0].toArray(),self.action_to_array(data[i][1]),self.action_to_array(data[i][2])])
            if(t < T):
                for j in range(len(old_traj)):
                    traj.append(old_traj[j])
                old_traj = traj
                t+= 1
            else: 
                t = 1
                old_traj = traj
                
            data_c.append(traj)
        
        #sort test training
        for i in range(len(data)):
            draw = np.random.uniform()
            if(draw >= 0.8): 
                self.test_tups.append(data_c[i])
            else: 
                self.train_tups.append(data_c[i])
        self.whole_data = data_c
        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def action_to_array(self,action): 
        array = np.zeros(5)
        array[action+1] = 1.0
        return array

    def array_to_action(self,action):
        v = np.argmax(action)
        return v -1

    def count_data(self,data, y_s):
        N = data.shape[0]
        current_density = np.zeros([self.h,self.w,self.T])
        current_count = np.zeros([self.h,self.w,self.T])
        labels = np.zeros([self.h,self.w])
        for i in range(N):
            x = data[i,0]
            y = data[i,1]
            t = data[i,3]
          
            current_density[x,y,t] = current_density[x,y,t] +data[i,2]
            labels[x,y] = self.array_to_action(y_s[i])

            current_count[x,y,t] = current_count[x,y,t] + 1.0
        
        density = np.zeros([self.h,self.w])
        for t in range(self.T):
            norm = np.sum(current_density[:,:,t])
            if(not norm == 0):
                current_density[:,:,t] = current_density[:,:,t]/norm
            density = density + current_density[:,:,t]
        

        density = density/np.sum(density)
  
        
        
        return density,labels
        
    def follow_the_leader(self):
        iters = len(self.densities)
        density = np.zeros([self.h,self.w])
        for i in range(iters):
            density = density+self.densities[i]

        density = density/np.sum(density)
        return density

    def get_weights(self,net,data = None):
        if(data == None):
            data = self.whole_data

        weights = np.zeros([len(data),4])
        y_s = []
        for i in range(len(data)):
            weights[i,2] = net.get_weight(data[i])
            weights[i,1] = data[i][0][0][1]
            weights[i,0] = data[i][0][0][0]
            weights[i,3] = len(data[i]) -1
            y_s.append(data[i][0][1])

        current_density,labels = self.count_data(weights,y_s)
        self.densities.append(current_density)
        #current_density = self.follow_the_leader()

        
        states = []
        for i in range(self.w):
            for j in range(self.h):
                a = (i,j)
                states.append( [a,self.action_to_array(labels[i,j])])
        states = np.array(states)

        return [states,current_density.flatten()]

    def next_w_train_batch(self,n,net,debug=False, resamp = False):

        if(resamp):
            self.states,self.dist = self.get_weights(net, data = self.train_tups)
        
        batch = np.random.choice(len(self.states),p=self.dist,size=n)
        batch = self.states[batch]


        if(debug and resamp):
            sample = np.random.choice(len(self.states),p=self.dist, size = 500)
            sample = self.states[sample]
            self.plotter.count_data(sample)
            self.plotter.show_states()
            self.plotter.plot_net_state_actions(net)

        weights = np.zeros([n,1])+1.0

        x = []
        y = []
        for i in range(len(batch)):
            x.append(batch[i][0])
            y.append(batch[i][1])
        
        return x,y, weights

    def next_ub_train_batch(self, n,net):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)

        batch = self.train_tups[self.i:n+self.i]
        x = []
        y = []
        for i in range(len(batch)):
            x.append(batch[i][0][0])
            y.append(batch[i][0][1])

        #Compute Weights

        #One set should be given
        #make fake ones equal 1
        weights = np.zeros([n,1])+1.0
        #One should be based on neural net

        weights = np.zeros([n,1])#+1.0
        for i in range(len(batch)):
            weights[i] = net.get_weight(batch[i])
            if(weights[i] >  1.0):
                print weights[i]
                print batch[i]
                IPython.embed()

        #print weights
        
        self.i = self.i + n
      


        return x, y, list(weights)
       



    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = self.test_tups
        x = []
        y = []
        for i in range(len(batch)):
            x.append(batch[i][0][0])
            y.append(batch[i][0][1])

        return x, y




class GridData(InputData):
    def __init__(self, data):
        self.train_tups = []
        self.test_tups = []
        self.i = 0
        data_c = []
        self.plotter = Plotter()
      
        #sort test training
        for i in range(len(data)):
            data_c.append([data[i][0].toArray(),self.action_to_array(data[i][1])])
            draw = np.random.uniform()
            if(draw >= 0.8): 
                self.test_tups.append(data_c[i])
            else: 
                self.train_tups.append(data_c[i])
    
        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def action_to_array(self,action): 
        array = np.zeros(5)
        array[action+1] = 1.0
        return array
    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)

        batch = self.train_tups[self.i:n+self.i]
        
        self.i = self.i + n
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])
       
  
       



    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = zip(*self.test_tups)
        return list(batch[0]), list(batch[1])

class IzzyData_B(InputData):
    def __init__(self, train_path, test_path,channels=3):
        self.train_tups = parse(train_path)
        self.test_tups = parse(test_path)

        self.i = 0
        self.channels = channels

        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)
        batch_tups = self.train_tups[self.i:n+self.i]
        batch = []
        for path, labels in batch_tups:
            im = cv2.imread(path)
         
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])


    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = []
        for path, labels in self.test_tups[:200]:
            im = cv2.imread(path,self.channels)
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        random.shuffle(self.test_tups)
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])


class AMTData(InputData):
    
    def __init__(self, train_path, test_path,channels=1):
        self.train_tups = parse(train_path)
        self.test_tups = parse(test_path)

        self.i = 0
        self.channels = channels

        random.shuffle(self.train_tups)
        random.shuffle(self.test_tups)

    def next_train_batch(self, n):
        """
        Read into memory on request
        :param n: number of examples to return in batch
        :return: tuple with images in [0] and labels in [1]
        """
        if self.i + n > len(self.train_tups):
            self.i = 0
            random.shuffle(self.train_tups)
        batch_tups = self.train_tups[self.i:n+self.i]
        batch = []
        for path, labels in batch_tups:
            im = cv2.imread(path)
         
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        batch = zip(*batch)
        self.i = self.i + n
        return list(batch[0]), list(batch[1])


    def next_test_batch(self):
        """
        read into memory on request
        :return: tuple with images in [0], labels in [1]
        """
        batch = []
        for path, labels in self.test_tups[:200]:
            im = cv2.imread(path,self.channels)
            im = im2tensor(im,self.channels)
            batch.append((im, labels))
        random.shuffle(self.test_tups)
        batch = zip(*batch)
        return list(batch[0]), list(batch[1])
