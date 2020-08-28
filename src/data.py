import numpy as np 
import os
import csv

def load_data(data_path):
    with open(data_path, 'r') as file:
        reader = list(csv.reader(file))
        
        attribute_names = (reader[0])
        attribute_names = attribute_names[:len(attribute_names)-1]
        
        arr = np.array(reader)
        arr = np.delete(arr, 0, 0)
        
        features = np.array([])
        targets = np.array([])
        
        
        for row in arr:
            targets = np.append(targets, float(np.array(row[len(row)-1])))
            temp_row=np.delete(row,int(len(row)-1))
            features = np.append(features,temp_row)
        
        new_features=np.empty([])
        
        for i in features:
            new_features=np.append(new_features, float(i))
        features=new_features

        features = np.delete(features, 0)
        features = features.reshape((np.size(arr,0),np.size(arr,1)-1))
        return (features, targets, attribute_names)

def train_test_split(features, targets, fraction):
    
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
        
    if (fraction==1.0):
        train_features=features
        train_targets=targets
        test_features=features
        test_targets=targets
        return train_features, train_targets, test_features, test_targets
    
    else:
        train_features=np.array([])
        train_targets=np.array([])
        test_features=np.array([])
        test_targets=np.array([])
        
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)

        features = features[indices]
        targets = targets[indices]
        
        index=int(fraction*features.shape[0])
        
        train_features=features[:index]
        test_features=features[index:]
        train_targets=targets[:index]
        test_targets=targets[index:]

        return train_features, train_targets, test_features, test_targets
