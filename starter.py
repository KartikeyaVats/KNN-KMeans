from turtle import distance
import k_nearest_neighbor, kmeans as km
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score

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
    k=6
    train_features = [rec[1] for rec in train]
    target = [[rec[0]] for rec in train]
    train_features = [list(map(float, sublist)) for sublist in train_features]
    train_features = np.array(train_features)
    train_features = train_features/255
    
    
    pca = PCA(n_components=29)
    pca.fit(train_features)
    train_features_pca = pca.transform(train_features)
    knn = k_nearest_neighbor.KNearestNeighbor(k,metric,"mode")
    knn.fit(train_features_pca, target)
    query_fetures = [rec[1] for rec in query]  # here query can be both validation or test set
    query_fetures = [list(map(float, sublist)) for sublist in query_fetures]
    query_fetures = np.array(query_fetures)
    query_fetures = query_fetures/255

    query_fetures_pca = pca.transform(query_fetures)
    labels = knn.predict(query_fetures_pca, False)
        
    return(labels.tolist())

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    labels = []
    sses = []
    k = 15
    #for k in range(1,100):
    train_features = [rec[1] for rec in train]
    train_features = [list(map(float, sublist)) for sublist in train_features]
    train_features = np.array(train_features)
    train_features = train_features/255
    #for nc in range(5,100):
    pca = PCA(n_components=5)
    pca.fit(train_features)
    train_features_pca = pca.transform(train_features)

    kmenas_obj = km.KMeans(k,metric)
    kmenas_obj.fit(train_features_pca)

    query_fetures = [rec[1] for rec in query]  # here query can be both validation or test set
    query_fetures = [list(map(float, sublist)) for sublist in query_fetures]
    query_fetures = np.array(query_fetures)
    query_fetures = query_fetures/255
    query_fetures_pca = pca.transform(query_fetures)
    labels, sse = kmenas_obj.predict(query_fetures_pca)
    # sses.append(sse)
    # print(nc)

    # plt.figure(figsize=(15,10))
    # plt.plot(range(5,100),sses, marker='o', markersize=9)
    # plt.savefig('./elbow_curve_sse_vs_k.png')
    print(silhouette_score(query_fetures, labels, metric='euclidean')) # does this work?
    return(labels.tolist())

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
    valid = read_data('./valid.csv') # used to tune hyperparameters
    #test = read_data('./test.csv')
    #y_pred = knn(train, test, "euclidean")
    #y_pred_flat = [rec[0] for rec in y_pred]
    #y_true = [rec[0] for rec in test]
    #print(y_pred_flat)
    #cm = confusion_matrix(y_true, y_pred)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    #plt.savefig('knn_test_cm.png')

    #print(labels)
    labels = kmeans(train, valid, "euclidean")
    print(labels)
    #show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    