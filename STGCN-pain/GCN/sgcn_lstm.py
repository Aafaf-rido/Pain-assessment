import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, GRU, Conv2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from sklearn.utils import check_array


tf.random.set_seed(2001)
np.random.seed(2001)

class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoach=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.lr = lr
        self.epoach = epoach
        self.batch_size = batch_size
        self.num_landmarks = 68

        self.INITIAL_LEARNING_RATE = lr
        self.DECAY_STEPS = 1000
        self.DECAY_RATE = 0.96

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.INITIAL_LEARNING_RATE,
            decay_steps=self.DECAY_STEPS,
            decay_rate=self.DECAY_RATE,
            staircase=True
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        # self.optimizer = Adam(learning_rate=self.INITIAL_LEARNING_RATE)

    def sgcn_gru(self, Input):
        """Temporal convolution"""
        k1 = Conv2D(64, (9, 1), padding='same', activation='relu')(Input)
        k = concatenate([Input, k1], axis=-1)

        """Graph Convolution"""

        """first hop localization"""
        x1 = Conv2D(64, kernel_size=(1, 1), strides=1, activation='relu')(k)
        x1 = Conv2D(68, kernel_size=(1, 1), strides=1, activation='relu')(x1)
        x_dim = Reshape(target_shape=(-1, x1.shape[2] * x1.shape[3]))(x1)
        f_1 = GRU(68, return_sequences=True)(x_dim)
        f_1 = tf.expand_dims(f_1, axis=3)
        logits = f_1
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])

        """Second hop localization"""
        y1 = Conv2D(64, kernel_size=(1, 1), strides=1, activation='relu')(k)
        y1 = Conv2D(68, kernel_size=(1, 1), strides=1, activation='relu')(y1)
        y_dim = Reshape(target_shape=(-1, y1.shape[2] * y1.shape[3]))(y1)
        f_2 = GRU(68, return_sequences=True)(y_dim)
        f_2 = tf.expand_dims(f_2, axis=3)
        logits = f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1])

        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

        """Temporal convolution"""
        z1 = Conv2D(16, (9, 1), padding='same', activation='relu')(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = Conv2D(16, (15, 1), padding='same', activation='relu')(z1)
        z2 = Dropout(0.25)(z2)
        z3 = Conv2D(16, (20, 1), padding='same', activation='relu')(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1)

        return z

    def Lstm(self, x):
        x = Reshape(target_shape=(-1, x.shape[2] * x.shape[3]))(x)
        rec = LSTM(15, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(20)(rec)
        rec1 = Dropout(0.25)(rec1)
        out = Dense(2, activation='softmax')(rec1)
        return out

    def train(self):
        seq_input = Input(shape=(self.train_x.shape[1], self.train_x.shape[2], self.train_x.shape[3]))

        x = self.sgcn_gru(seq_input)
        y = self.sgcn_gru(x)
        y = y + x
        z = self.sgcn_gru(y)
        z = z + y

        out = self.Lstm(z)
        self.model = Model(seq_input, out)

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        # Ensure train_y is a numpy array of integers
        self.train_y = np.asarray(self.train_y, dtype=np.int32)

        # Check train_y for valid classification targets
        check_array(self.train_y, ensure_2d=False)
        if len(np.unique(self.train_y)) > 2:
            raise ValueError("The number of classes in train_y is greater than 2. Please check the labels.")

        # Reshape train_x to (num_samples, num_frames * num_landmarks * num_features)
        num_samples, num_frames, num_landmarks, num_features = self.train_x.shape
        X_reshaped = self.train_x.reshape(num_samples, num_frames * num_landmarks * num_features)

        # Apply SMOTE
        smote = SMOTE()
        X_smote, y_smote = smote.fit_resample(X_reshaped, self.train_y)

        # Reshape X_smote back to (num_samples, num_frames, num_landmarks, num_features)
        X_smote = X_smote.reshape(-1, num_frames, num_landmarks, num_features)

        # Define the ModelCheckpoint callback
        # checkpoint = ModelCheckpoint(
        #     "sgcn.h5", save_best_only=True, monitor="val_accuracy", mode="min"
        # )

        history = self.model.fit(
            X_smote,
            y_smote,
            validation_split=0.1,
            epochs=self.epoach,
            batch_size=self.batch_size
            # callbacks=[checkpoint]
        )

        return history

    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred




