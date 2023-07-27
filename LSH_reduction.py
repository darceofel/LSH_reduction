import copy
import math
import numpy as np
from scipy import stats
import torch

cos_sim = lambda a,b: np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

class Random_projection_hash():
    def __init__(self, trainset):
        length = trainset.data.shape[0]
        ## Turn dataset into a numpy array
        if isinstance(trainset.data, np.ndarray):
            self.array = trainset.data.reshape(length, -1).astype('float')
        elif isinstance(trainset.data, torch.Tensor):
            self.array = trainset.data.numpy().reshape(length, -1).astype('float') 
        
        self.trainset_array = self.array - self.array.mean(axis = 0) ## Re-center the dataset array
        self.d = self.array.shape[1] ## Save dimension of data points

        self.ix_to_label = [] ## digit by index in the dataset
        self.label_to_ixs = {} ## indexes that represent each digit
        self.label_to_reps = {} ## How many repetitions by label

        for ix, (vec, label) in enumerate(trainset):
            self.ix_to_label.append(label)
            
            self.label_to_ixs[label] = self.label_to_ixs.get(label, [])
            self.label_to_ixs[label].append(ix)
            
            self.label_to_reps[label] = self.label_to_reps.get(label, 0) + 1

        self.num_labels = len(self.label_to_ixs)

    def hash_values(self, nbits = None, n_hashtables = None):
        if not nbits:
            nbits = int(math.log2(self.array.shape[0])/2) + 1
        if not n_hashtables:
            n_hashtables = min(int(self.num_labels**(1/2)), int(self.d/nbits))
        print(f'Using: {nbits} bits, {n_hashtables} hash tables') 

        rand_tables = np.random.normal(0,1,(n_hashtables, nbits, self.d))
        
        ## save which indexes have each hash for each hash table
        self.tb_inthash_ix = [{} for i in range(n_hashtables)] 
        self.n_hashtables = n_hashtables

        binary_array = (np.matmul(rand_tables, self.trainset_array.T) > 0).astype(int).transpose(0,2,1)
        ## Instead of saving hashes as binary strings, we save the integer it represents
        self.hashes_array = binary_array.dot(np.flip(1<<np.arange(nbits))).T

        for ix in range(self.trainset_array.shape[0]):
            hash = self.hashes_array[ix]
            label = self.ix_to_label[ix]
            for i in range(n_hashtables):
                self.tb_inthash_ix[i][hash[i]] = self.tb_inthash_ix[i].get(hash[i], {lab:[] for lab in self.label_to_ixs.keys()})
                self.tb_inthash_ix[i][hash[i]][label].append(ix)

    def find_most_alike(self, ix):
        label = self.ix_to_label[ix]
        hash = self.hashes_array[ix]

        # Return indexes that share the same hash on a hash table and share the same label
        return set([j for i in range(self.n_hashtables) for j in self.tb_inthash_ix[i][hash[i]][label]]) - set([ix])

    def estimation(self):
        # This fucntion estimates the mean and std of the distance
        # of points to its neighbors by label
        est_mean = {}
        est_var = {}
        
        # Statistical constants
        Z = 1.65
        E = 0.05
        p = 0.5

        for label, N in self.label_to_reps.items():
            # Determine sample size
            n = int(((Z**2 * N)*(p * (1-p))) / (((E**2)*(N-1)) + ((Z**2)*(p *(1-p))))) 
            curr_sample = np.random.choice(self.label_to_ixs[label], n)
            
            distances_lst = [] # Save distances to estimate their mean and std
            for ix in curr_sample:
                vec = self.array[ix]
                distances_lst.append(np.array([1-cos_sim(vec, self.array[i]) for i in self.find_most_alike(ix)]).mean())
            
            distances_lst = np.array(distances_lst)
            est_mean[label] = np.nanmean(distances_lst)
            est_var[label] = np.nanstd(distances_lst)**2

        self.est_mean = est_mean
        self.est_var = est_var
        
def shrink(rp, p):
    # Renaming
    vectors = rp.array
    ixs_by_label = copy.deepcopy(rp.label_to_ixs)

    deletion_by_label = {label:int(n*p) for label, n in rp.label_to_reps.items()}

    chosen = []
    rp.label_to_th = {}
    for label in ixs_by_label.keys():
        # Get the estimations of mean and variance
        m = rp.est_mean[label]
        v = rp.est_var[label]
        
        # Estimate the p-percentile using a Beta distribution
        alpha = (m**2- m**3 - m*v)/v 
        beta = (alpha*v*(alpha + 1))/(m**2-alpha*v)
        th = stats.beta.ppf(p, alpha, beta) # define the threshold for this label as that percentile
        rp.label_to_th[label] = th
        
        current_ixs = ixs_by_label[label]
        label_max_deletion = deletion_by_label[label]

        count = 0
        while count < label_max_deletion:
            try:
                ix = np.random.choice(current_ixs)
            except(ValueError):
                print(f"The algorithm reached the limit of elimination of {label}'s")
                break

            chosen.append(ix)
            curr_vector = vectors[ix]
            current_ixs.remove(ix)

            alike_neighbours = rp.find_most_alike(ix)
            ## Sort indexes of neighbors by distance to the current chosen one
            neighbours_dist = sorted([(i, 1-cos_sim(vectors[i], curr_vector)) for i in alike_neighbours], key = lambda x: x[1])

            local_count = 0
            local_max = int(len(alike_neighbours)*p)
            for alike_ix, distance in neighbours_dist:
                ## Eliminate all of the neighbors that are closer than the threshold
                if count >= label_max_deletion or local_count >= local_max or distance > th:
                    break
                try:
                    current_ixs.remove(alike_ix)
                    count += 1
                    local_count += 1
                except(ValueError):
                    continue
    
    ## Return all the chosen indexes and the indexes that weren't eliminated
    return chosen + [i for label in ixs_by_label.keys() for i in ixs_by_label[label]]


