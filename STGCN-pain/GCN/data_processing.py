import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from IPython.core.debugger import set_trace



class Data_Loader():
    def __init__(self, dir):
        # self.num_repitation = 5
        self.num_channel = 2 #Modified (x,y)
        self.dir = dir
        # self.body_part = self.body_parts()
        self.num_landmarks = 68       
        self.dataset = []
        self.sequence_length = []
        self.num_frames = 20 # self.num_timestep = 100
        
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        

        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()
                
    
   

    def import_dataset(self):
        train_x = pd.read_csv("./" + self.dir+"/Train_X.csv").iloc[:,:].values # check what dir is!
        train_y = pd.read_csv("./" + self.dir+"/Train_Y.csv").iloc[:,:].values
        return train_x, train_y

    def augment_data(self, X, y):
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            augmented_X.append(X[i])
            augmented_y.append(y[i])

            augmented_X.append(self.random_translate(X[i]))
            augmented_y.append(y[i])

            augmented_X.append(self.random_rotate(X[i]))
            augmented_y.append(y[i])

            # augmented_X.append(self.random_noise(X[i]))
            # augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)

    def random_translate(self, sequence):
        # Translation aléatoire 
        max_translation = 5
        translated_sequence = sequence + np.random.uniform(-max_translation, max_translation, sequence.shape)
        return translated_sequence

    def random_rotate(self, sequence):
        # Rotation aléatoire 
        max_angle = np.pi / 18
        angle = np.random.uniform(-max_angle, max_angle)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        
        # Appliquer la rotation uniquement sur les coordonnées x et y
        rotated_sequence = np.copy(sequence)
        for t in range(sequence.shape[0]):
            for j in range(sequence.shape[1]):
                xy = sequence[t, j, :2]  # Extraire les coordonnées x et y
                rotated_xy = np.dot(xy, rotation_matrix)  # Appliquer la rotation
                rotated_sequence[t, j, :2] = rotated_xy  # Mettre à jour les coordonnées x et y
        
        return rotated_sequence

    def random_noise(self, sequence):
        # Ajout de bruit gaussien
        noise = np.random.normal(0, 0.1, sequence.shape)
        noisy_sequence = sequence + noise
        return noisy_sequence

    
           
    def preprocessing(self):
        # turn the data into columns=(x1,y1,x2,y2,...,x68,y68)
        X_train = np.zeros((self.train_x.shape[0],self.num_landmarks*self.num_channel)).astype('float32')
        for row in range(self.train_x.shape[0]):
            counter = 0
            for landmark in range(self.train_x.shape[1]):
                for i in range(self.num_channel):
                    # print(type(self.train_x[0][0]))

                    landmarks = self.train_x[row][landmark]
                    landmarks = landmarks.strip('()').split(',')

                    X_train[row, counter+i] = landmarks[i]
                
                counter += self.num_channel 
        
           
        y_train = np.reshape(self.train_y,(-1,1))
        X_train = self.sc1.fit_transform(X_train)   
   
        # y_train = self.sc2.fit_transform(y_train)  
        
        print(y_train.shape[0])
        for i in range(y_train.shape[0]):
            if y_train[i] == 'nopain':
                y_train[i] = 0
            else:
                y_train[i] = 1


        print(X_train.shape)
        X_train_ = np.zeros((self.batch_size, self.num_frames, self.num_landmarks, self.num_channel))
        
        for batch in range(X_train_.shape[0]):
            for frame in range(X_train_.shape[1]):
                for landmark in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch,frame,landmark,channel] = X_train[frame+(batch*self.num_frames),channel+(landmark*self.num_channel)]
            
                        
        X_train = X_train_                
        return X_train, y_train
