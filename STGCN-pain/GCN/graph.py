import tensorflow as tf
import numpy as np
from IPython.core.debugger import set_trace

class Graph():
    def __init__(self, num_node):
        self.num_node = num_node
        self.AD, self.AD2, self.bias_mat_1, self.bias_mat_2 = self.normalize_adjacency()
        
    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)]

        neighbor_1base = [(0,36), (0,1), (0,17),
                          (1,2), (1,36), (1,41),
                          (2,3), (2,31), (2,41),
                          (3,4), (3,31), (3,48),
                          #12
                          #30
                          (4,5), (4,48), 
                          (5,6), (5,48), 
                          (6,7), (6,48), (6,59),
                          (7,8), (7,58), (7,59),
                          (8,9), (8,56), (8,57), (8,58),
                          (9,10),(9,55),(9,56),
                          (10,11),(10,54),(10,55),
                          (11,12),(11,54),
                          (12,13),(12,54),
                          (13,14),(12,35),(13,54),
                          (14,15),(14,35),(14,46),
                          #30
                          (15,16),(15,45),(15,46),
                          (16,26),(16,45),
                          (17,18),(17,36),
                          (18,19),(18,36),(18,37),
                          (19,20),(19,37),(19,38),
                          (20,21),(20,38),(20,39),(20,23),
                          (21,22),(21,23),(21,27),(21,39),
                          (22,23),(22,27),(22,42),
                          (23,24),(23,42),(23,43),
                          (24,25),(24,43),(24,44),
                          #20
                          (25,26),(25,44),(25,45),
                          (26,45),
                          (27,28),(27,39),(27,42),
                          (28,29),(28,39),(28,42),
                          (29,30),(29,39), (29,40), (29,42),(29,47),
                          (30,31),(30,32),(30,33),(30,34),(30,35),
                          #21
                          (31,32),(31,40),(31,41),(31,48),(30,49),(31,50),
                          (32,33),(32,50),(32,51),
                          (33,34),(33,51),
                          (34,35),(34,51),(34,52),
                          (35,46),(35,47),(35,52),(35,53),(35,54),
                          (36,37),(36,41),
                          #22
                          (37,38),(37,40),(37,41),
                          (38,39),(38,40),
                          (39,40),
                          (40,41),
                          (42,43),(42,47),
                          (43,44),(43,47),
                          (44,45),(44,46),(44,47),
                          (45,46),
                          (46,47),
                          (48,49),(48,59),(48,60),
                          (49,50),(49,60),(49,61),
                          #19
                          (50,51),(50,61),(50,62),
                          (51,52),(51,62),
                          (52,53),(52,62),(52,63),
                          (53,54),(53,63),(53,64),
                          (54,55),(54,64),
                          (55,56),(55,64),(55,65),
                          (56,57),(56,65),(56,66),
                          #20
                          (57,58),(57,66),
                          (58,59),(58,66),(58,67),
                          (59,40),(59,67),
                          (60,61),(60,67),
                          (61,62),(61,66),(61,67),
                          (62,63),(62,66),
                          (63,64),(63,65),(63,66),
                          (64,65),
                          (65,66),
                          (66,67)
                         # total 174                                                  
                          ]

        
    

        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link    
        A = np.zeros((self.num_node, self.num_node)) # adjacency matrix
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        
        A2 = np.zeros((self.num_node, self.num_node)) # second order adjacency matrix
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root,neighbour_of_neigbour] = 1         

        #AD = self.normalize(A)
        #AD2 = self.normalize(A2)
        bias_mat_1 = np.zeros(A.shape)
        bias_mat_2 = np.zeros(A2.shape)
        bias_mat_1 = np.where(A!=0, bias_mat_1, -1e9)
        bias_mat_2 = np.where(A2!=0, A2, -1e9)
        AD = A.astype('float32')
        AD2 = A2.astype('float32')
        bias_mat_1 = bias_mat_1.astype('float32')
        bias_mat_2 = bias_mat_2.astype('float32')
        AD = tf.convert_to_tensor(AD)
        AD2= tf.convert_to_tensor(AD2)
        bias_mat_1 = tf.convert_to_tensor(bias_mat_1)
        bias_mat_2 = tf.convert_to_tensor(bias_mat_2)
        print("----------------------------")
        print(bias_mat_1.shape)
        print(bias_mat_2.shape)
        print("----------------------------")
       
        return AD, AD2, bias_mat_1, bias_mat_2
        
    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype('float32')
        normalize_adj = tf.convert_to_tensor(normalize_adj)   
        return normalize_adj
