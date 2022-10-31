from turtle import distance
import k_nearest_neighbor, kmeans as km
import numpy as np
from scipy import spatial

def accuracy_knn(labels, gt_labels):
    correct = 0
    for i in range(len(labels)):
        if labels[i] == gt_labels[i]:
            correct+=1
    return correct/len(gt_labels)


def plot_accuracy_curve(acc_map):
    for k,v in acc_map.items():
        plt.plot(k, v, label=k)
    plt.savefig('./accuracy_curve.png')

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    dist = np.linalg.norm(a-b)
    return(dist)
        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    #dist = 1 - spatial.distance.cosine(a, b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    dist = np.dot(a,b)/(a_norm*b_norm)
    return(1-dist)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    k=1
    train_features = [rec[1] for rec in train]
    target = [[rec[0]] for rec in train]
    train_features = [list(map(float, sublist)) for sublist in train_features]
    train_features = np.array(train_features)
    
    knn = k_nearest_neighbor.KNearestNeighbor(k,metric,"mode")
    knn.fit(train_features, target)

    query_fetures = [rec[1] for rec in query]  # here query can be both validation or test set
    query_fetures = [list(map(float, sublist)) for sublist in query_fetures]
    query_fetures = np.array(query_fetures)

    labels = knn.predict(query_fetures, False)


    return(labels.tolist())

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    labels = []
    k = 10
    train_features = [rec[1] for rec in train]
    train_features = [list(map(float, sublist)) for sublist in train_features]
    train_features = np.array(train_features)
    kmenas_obj = km.KMeans(k,metric)
    kmenas_obj.fit(train_features)

    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    train = read_data('./train.csv')
    valid = read_data('./valid.csv')
    #labels = knn(train, valid, "cosine")
    #print(labels)
    labels = kmeans(train, valid, "euclidean")
    #show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    