import numpy as np

def confusion_matrix(actual, predictions):

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    TP=0
    TN=0
    FP=0
    FN=0
    
    for i in range(len(actual)):
        if (actual[i]==1):
            if (predictions[i]==1):
                TP+=1
            else:
                FN+=1
        else:
            if (predictions[i]==1):
                FP+=1
            else:
                TN+=1
    confusion_matrix=np.array([[TN,FP],[FN,TP]])
    return confusion_matrix

def accuracy(actual, predictions):
    
    ans=confusion_matrix(actual, predictions)
    if ((ans[1][1]+ans[0][0]+ans[0][1]+ans[1][0])==0):
        return 0
    else:
        return (ans[1][1]+ans[0][0])/(ans[1][1]+ans[0][0]+ans[0][1]+ans[1][0])
    
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")


def precision_and_recall(actual, predictions):
    
    ans=confusion_matrix(actual, predictions)
    if ((ans[1][1]+ans[0][1])==0):
        first=0
    else:
        first=ans[1][1]/(ans[1][1]+ans[0][1])
    if((ans[1][1]+ans[1][0])==0):
        second=0
    else:
        second=ans[1][1]/(ans[1][1]+ans[1][0])
    return first, second
    
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")


def f1_measure(actual, predictions):
    
    precision, recall= precision_and_recall(actual, predictions)
    if(precision+recall==0):
        return 0
    else:
        return 2*(precision*recall)/(precision+recall)
    
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

