import numpy as np
import random
#from starter import euclidean, cosim


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

def sse(means, clusters, assigned_means, features):
    assigned_means = np.array(assigned_means)
    means = np.array(means)
    total = 0
    for cluster in range(clusters):
        indexes = np.where(assigned_means==cluster)[0].tolist()
        data_clusterd = features[indexes]
        sq_diff = (means[cluster] - data_clusterd)**2
        sq_diff_sum = np.sum(sq_diff, axis=1)
        err = np.sum(sq_diff_sum**0.5, axis=0)#np.sum(np.sum(((means[cluster] - data_clusterd)**2)**0.5, axis=1))
        total+=err
    return total

class KMeans():
    def __init__(self, n_clusters, distance_metric):  #added the distance parameter, can we change stuff?
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None
        self.distnce_metric = distance_metric

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        
        self.means = random.sample(features.tolist(), self.n_clusters)
        while(1):
            distances = []
            assigned_means = []
            means = self.means.copy()
            for data in features:
                for mean in self.means:
                    if self.distnce_metric=='euclidean':
                        distances.append(euclidean(data,mean))
                    else:
                        distances.append(cosim(data,mean))
                min_dist_mean = sorted(range(len(distances)), key=distances.__getitem__)[:1]
                assigned_means.append(min_dist_mean[0])
                distances = []

            assigned_means = np.array(assigned_means)
            for cluster in range(self.n_clusters):
                indexes = np.where(assigned_means==cluster)[0].tolist()
                data_clusterd = features[indexes]
                #print(np.mean(data_clusterd))
                means[cluster] = np.mean(data_clusterd, axis=0)
            diff = [[a==b for a,b in zip(la,lb)] for la,lb in zip(means,self.means)]
            diff_flatten = [item for sublist in diff for item in sublist]
            if not all(diff_flatten):
                self.means = means
            else:
                #input(len(data_clusterd))
                break
        return None


    

        #raise NotImplementedError()

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        distances = []
        assigned_means = []
        #input(len(self.means))
        for feature in features:
            for mean in self.means:
                if self.distnce_metric == 'euclidean':
                    dist = euclidean(feature, mean)
                else:
                     dist = cosim(feature, mean)
                distances.append(dist)
            min_dist_mean = sorted(range(len(distances)), key=distances.__getitem__)[:1]
            assigned_means.append(min_dist_mean[0])
            distances = []
        error = sse(self.means, self.n_clusters, assigned_means, features)
        return np.array(assigned_means), error

        #raise NotImplementedError()