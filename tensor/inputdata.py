import random
import numpy as np
import IPython
import tensorflow as tf
import cv2
import IPython

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

class GridData(InputData):
    def __init__(self, data):
        self.train_tups = []
        self.test_tups = []
        self.i = 0
        data_c = []
      
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
