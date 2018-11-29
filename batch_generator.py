import gesture_processing as gp
import numpy as np
import pandas as pd
from keras.utils import to_categorical

train = pd.read_csv('train-swipes.csv',index_col = 0)
train['label_no'] = train['gesture'].apply(gp.label2num)


#generates a batch of 100 samples

image_shape = (32,32)

def batch_gen():
    n = 0
    while True:
        X = np.zeros((100,40) + image_shape+(1,))
        Y_labels = np.zeros(100)
        for i in range(100):
            X[i] = gp.gest3D(train.iloc[i+n*100,0]).reshape(40,image_shape[0],image_shape[1],1)
            Y_labels[i] = train.iloc[i+n*100,2]

        Y = to_categorical(Y_labels,4)
        n += 1
        yield (X,Y)
        


# generates an array of gesture samples from the training set

def sample_gen(sample_no):
    result = np.zeros((sample_no,40,image_shape[0],image_shape[1],1))
    for i in range(sample_no):
        result[i] = gp.gest3D(train.iloc[i,0]).reshape(40,image_shape[0],image_shape[1],1)
    return result

# returns a tuple of test targets and gesture samples

def test_gen():
    test = pd.read_csv('test-swipes.csv')
    test['label_no'] = test['gesture'].apply(gp.label2num)
    gestures = np.zeros((2008,40,32,32,1))
    targets = np.zeros((2008,4))
    from keras.utils import to_categorical
    targets = to_categorical(test['label_no'].values,4)

    for i in range(2008):
        gestures[i] = gp.gest3D(test.iloc[i,1]).reshape(40,32,32,1)

    return gestures, targets


