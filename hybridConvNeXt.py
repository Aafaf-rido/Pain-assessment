import os
import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns


def main():
    with tf.device('/device:GPU:0'):
        tf.random.set_seed(2001)
        np.random.seed(2001)
        extract_features = False

        folder_path = r"/PEMF/Pictures/Modified"
        
        train_features = r"convnextXL_features_data\train_features.npy"
        train_labels = r"convnextXL_features_data\train_labels.npy"
        val_features = r"convnextXL_features_data\val_features.npy"
        val_labels = r"convnextXL_features_data\val_labels.npy"
        test_features = r"convnextXL_features_data\test_features.npy"
        test_labels = r"convnextXL_features_data\test_labels.npy"

        IMG_SIZE = 224
        NUM_FRAMES = 20
        NUM_FEATURES = 2048

        EPOCHS=100
        BATCH_SIZE = 8

        INITIAL_LEARNING_RATE = 0.001
        DECAY_STEPS = 1000
        DECAY_RATE = 0.96


        if extract_features:
            # Get data paths
            def get_data(folder_path):
                video_paths = []
                labels = []
                df = pd.DataFrame(columns=['video_path', 'tag'])

                for subject in os.listdir(folder_path):
                    subject_path = os.path.join(folder_path, subject)
                    if os.path.isdir(subject_path):
                        for stimulus in os.listdir(subject_path):
                            stimulus_path = os.path.join(subject_path, stimulus)

                            if os.path.isdir(stimulus_path):
                                # if stimulus == 'Neutral' or stimulus == 'Posed Pain':
                                for folder in os.listdir(stimulus_path):
                                    frames_path = os.path.join(stimulus_path, folder)
                                    video_paths.append(frames_path)
                                    
                                    if 'Neutral' in stimulus:
                                        label = 'Nopain'
                                    elif 'Posed Pain' in stimulus:
                                        label = 'Pain'
                                    elif 'Algometer Pain' in stimulus:
                                        label = 'Pain'
                                    else:
                                        label = 'Pain'

                                    labels.append(label)
            

                df['video_path'] = video_paths
                df['tag'] = labels

                return df



            data = get_data(folder_path)

            print(data.head(5))
            print(len(data['video_path']))
            print(data["tag"].value_counts())   

            # Spliting the data into test, train  and validation sets
            train_df, test_df = train_test_split(data, test_size=0.1, random_state=42, stratify=data['tag'])
            train_df, val_df = train_test_split(train_df, test_size=0.1)



            print('---------TRAIN-------------', train_df.shape)
            print('---------test-------------', test_df.shape)
            print('---------VAL--------------', val_df.shape)


            # Get the class names
            label_processor = keras.layers.StringLookup(
                num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
            )

            print(label_processor.get_vocabulary())


            # Load frames 
            def load_frames_from_folder(folder_path):
                frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                frames = []
                for frame_file in frame_files:
                    frame = cv2.imread(frame_file)

                    if frame is not None:
                        # Check if the image is grayscale
                        if len(frame.shape) == 2 or frame.shape[2] == 1:
                            frame = np.stack((frame,) * 3, axis=-1)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                        frame = tf.cast(frame, tf.float32) / 255.0
    
                        # Standardize the image by subtracting the mean and dividing by the standard deviation
                        frame = (frame - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])

                        frames.append(frame)
                        if len(frames) == NUM_FRAMES:
                            break
                    else:
                        print(f"Could not read frame: {frame_file}")
                if len(frames) != NUM_FRAMES:
                    print(f"Folder {folder_path} does not have enough frames. Only {len(frames)} were loaded.")
                return np.array(frames)



            
            def generate_random_params():
                params = {
                    "flip": tf.random.uniform([], 0, 1) > 0.5,
                    "brightness": tf.random.uniform([], 0.0, 0.4),  # Ensure non-negative brightness adjustment
                    "contrast": tf.random.uniform([], 0.8, 1.2),
                    "saturation": tf.random.uniform([], 0.8, 1.2),
                    "hue": tf.random.uniform([], 0.0, 0.1),  # Ensure non-negative hue adjustment
                    "apply_affine": tf.random.uniform([], 0, 1) > 0.3,
                    "scale": tf.random.uniform([], 0.8, 1.0),
                    "translate": tf.random.uniform(shape=[2], minval=-0.2, maxval=0.2),
                    "rotation": tf.random.uniform([], -15, 15),
                    "apply_erasing": tf.random.uniform([], 0, 1) > 0.5,
                }
                return params
            

            def augment_frame(frame, params):
                # Ensure the frame has the correct shape
                if len(frame.shape) == 3:
                    frame = tf.expand_dims(frame, axis=0)
                
                if params["flip"]:
                    frame = tf.image.flip_left_right(frame)
                frame = tf.image.adjust_brightness(frame, params["brightness"])  # Adjust brightness
                frame = tf.image.random_contrast(frame, lower=0.8, upper=params["contrast"])
                frame = tf.image.random_saturation(frame, lower=0.8, upper=params["saturation"])
                frame = tf.image.random_hue(frame, max_delta=params["hue"])
                if params["apply_affine"]:
                    frame = tfa.image.transform(frame, [1, 0, params["translate"][0] * IMG_SIZE, 0, 1, params["translate"][1] * IMG_SIZE, 0, 0])
                    frame = tfa.image.rotate(frame, params["rotation"] * (np.pi / 180))
                crop_size = tf.random.uniform([], minval=int(IMG_SIZE * params["scale"]), maxval=IMG_SIZE, dtype=tf.int32)
                frame = tf.image.resize_with_crop_or_pad(frame, crop_size, crop_size)
                frame = tf.image.resize(frame, [IMG_SIZE, IMG_SIZE])
                
                if params["apply_erasing"]:
                    erase_area = tf.random.uniform([], 0.02, 0.4) * IMG_SIZE * IMG_SIZE
                    erase_aspect_ratio = tf.random.uniform([], 0.3, 3.3)
                    erase_height = tf.sqrt(erase_area / erase_aspect_ratio)
                    erase_width = erase_area / erase_height
                    
                    # Ensure erase_height and erase_width are even
                    erase_height = tf.cast(tf.math.ceil(erase_height / 2) * 2, tf.int32)
                    erase_width = tf.cast(tf.math.ceil(erase_width / 2) * 2, tf.int32)
                    
                    # Ensure erase_height and erase_width do not exceed image dimensions
                    erase_height = tf.minimum(erase_height, IMG_SIZE)
                    erase_width = tf.minimum(erase_width, IMG_SIZE)
                    
                    frame = tfa.image.random_cutout(frame, [erase_height, erase_width], constant_values=0)
                
                # Squeeze the frame back to remove the batch dimension if it was added
                if frame.shape[0] == 1:
                    frame = tf.squeeze(frame, axis=0)
                
                return frame



            def augment_frames(frames):
                params = generate_random_params()  # Generate random parameters for a single clip
                augmented_frames = [augment_frame(frame, params) for frame in frames]
                return np.array(augmented_frames)
            

            

            def save_features_labels(features, labels, features_path, labels_path):
                # Ensure the directory exists
                features_dir = os.path.dirname(features_path)
                labels_dir = os.path.dirname(labels_path)
                
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)
                
                if not os.path.exists(labels_dir):
                    os.makedirs(labels_dir)
                
                # Save features and labels
                np.save(features_path, features)
                np.save(labels_path, labels)

            def prepare_all_videos(df, features_path, labels_path, augment=False):
                num_samples = len(df)
                video_paths = df["video_path"].values.tolist()
                labels = df["tag"].values
                labels = tf.convert_to_tensor(label_processor(labels[..., None])).numpy()

                # Estimate the expanded dataset size
                expanded_size = num_samples * 2 if augment else num_samples
                counter = 0
                # Pre-allocate array for frame features
                frame_features = np.zeros((expanded_size, NUM_FRAMES, NUM_FEATURES), dtype="float32")
                all_labels = []

                idx = 0
                for path, label in zip(video_paths, labels):
                    # print(path)
                    frames = load_frames_from_folder(path)

                    if frames.shape[0] != NUM_FRAMES:
                        print(f"Video {path} does not have enough frames.")
                        continue

                    # Augment frames for expanded dataset
                    if augment:
                        aug_frames = augment_frames(frames)
                        for j in range(NUM_FRAMES):
                            if idx >= expanded_size:
                                print(f"Index {idx} is out of bounds for frame_features with size {expanded_size}.")
                                break
                            frame_features[idx, j, :] = feature_extractor.predict(aug_frames[None, j, :], verbose=0)
                        all_labels.append(label)
                        idx += 1

                    for j in range(NUM_FRAMES):
                        if idx >= expanded_size:
                            print(f"Index {idx} is out of bounds for frame_features with size {expanded_size}.")
                            break
                        frame_features[idx, j, :] = feature_extractor.predict(frames[None, j, :], verbose=0)
                    all_labels.append(label)
                    idx += 1

                    print(f'Features extracted for video {counter} out of {num_samples}')
                    counter += 1

                frame_features = frame_features[:idx]
                all_labels = np.array(all_labels)

                save_features_labels(frame_features, all_labels, features_path, labels_path)

                return frame_features, all_labels




            def build_feature_extractor():
                feature_extractor = keras.applications.ConvNeXtXLarge(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                )

                inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                print(inputs.shape)
                outputs = feature_extractor(inputs)
                return tf.keras.Model(inputs, outputs, name="feature_extractor")       


            feature_extractor = build_feature_extractor()
            # Prepare training, validation, and test sets
            val_data, val_labels = prepare_all_videos(val_df, val_features, val_labels, augment=False)
            train_data, train_labels = prepare_all_videos(train_df, train_features, train_labels, augment=True)
            test_data, test_labels = prepare_all_videos(test_df, test_features, test_labels, augment=False)

    
        else: 
            # Load features and labels from disk
            def load_features_labels(features_path, labels_path):
                features = np.load(features_path)
                labels = np.load(labels_path)
                return features, labels

            train_data, train_labels = load_features_labels(train_features, train_labels)
            val_data, val_labels = load_features_labels(val_features, val_labels)
            test_data, test_labels = load_features_labels(test_features, test_labels)

            # Ensure all labels are strings
            all_labels = np.concatenate([train_labels, val_labels, test_labels]).astype(str)

            label_processor = keras.layers.StringLookup(
                num_oov_indices=0, vocabulary=np.unique(all_labels)
            )
    

            print(label_processor.get_vocabulary())


        print(f" train data shape: {train_data.shape}, train labels shape: {train_labels.shape}")
        print(f" test data shape: {test_data.shape}, test labels shape: {test_labels.shape}")
        print(f" validation data shape: {val_data.shape}, validation labels shape: {val_labels.shape}")




        # Compute class weights
        def compute_class_weights(labels):
            labels_flat = labels.flatten()
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels_flat),
                y=labels_flat
            )
            class_weight_dict = dict(enumerate(class_weights))
            return class_weight_dict

        class_weights = compute_class_weights(train_labels)
        print("Computed class weights:", class_weights)


        
        

        def build_cnn_lstm_model(num_classes=2):
            cnn_input = tf.keras.Input(shape=(NUM_FRAMES, NUM_FEATURES))
            lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(cnn_input)
            lstm_out = tf.keras.layers.LSTM(64)(lstm_out)
            x = tf.keras.layers.Dense(128, activation='relu')(lstm_out)
            output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs=cnn_input, outputs=output)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=INITIAL_LEARNING_RATE,
                decay_steps=DECAY_STEPS,
                decay_rate=DECAY_RATE,
                staircase=True)

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["accuracy"]
            )
            return model


        # Bi-LSTM
        def build_cnn_Bilstm_model(num_classes=2):
            cnn_input = tf.keras.Input(shape=(NUM_FRAMES, NUM_FEATURES))
            bilstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(cnn_input)
            bilstm_out = tf.keras.layers.Bidirectional (tf.keras.layers.LSTM(64))(bilstm_out)
            x = tf.keras.layers.Dense(128, activation='relu')(bilstm_out)
            output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs=cnn_input, outputs=output)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=INITIAL_LEARNING_RATE,
                decay_steps=DECAY_STEPS,
                decay_rate=DECAY_RATE,
                staircase=True)

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["accuracy"]
            )
            return model

        # Stacked LSTM + Att
        def build_cnn_stacked_lstm_att_model(num_classes=2):
            cnn_input = tf.keras.Input(shape=(NUM_FRAMES, NUM_FEATURES))
            lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(cnn_input)
            lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(lstm_out)
            lstm_out = tf.keras.layers.LSTM(64)(lstm_out)

            # Attention Layer
            attention = tf.keras.layers.Attention()([lstm_out, lstm_out])
            
            # Flatten the attention output to match the shape expected by Dense layers
            attention = tf.keras.layers.Flatten()(attention)
            
            x = tf.keras.layers.Dense(128, activation='relu')(lstm_out)
            x = tf.keras.layers.Dropout(rate=0.25)(x)
            output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
            model = tf.keras.Model(inputs=cnn_input, outputs=output)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=INITIAL_LEARNING_RATE,
                decay_steps=DECAY_STEPS,
                decay_rate=DECAY_RATE,
                staircase=True)

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["accuracy"]
            )
            return model


        # LSTM with attention
        def build_cnn_lstm_model_with_attention(num_classes=2):
            # CNN input layer for sequence data
            cnn_input = tf.keras.Input(shape=(NUM_FRAMES, NUM_FEATURES))
            
            # LSTM layers
            lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(cnn_input)
            
            # Attention mechanism
            query = lstm_out
            value = lstm_out
            attention_out, attention_scores = tf.keras.layers.Attention()([query, value], return_attention_scores=True)
            
            # Another LSTM layer after attention
            lstm_out = tf.keras.layers.LSTM(64)(attention_out)
            
            # Dense layers for classification
            x = tf.keras.layers.Dense(128, activation='relu')(lstm_out)
            output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            # Define the model
            model = tf.keras.Model(inputs=cnn_input, outputs=output)
            
            # Learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=INITIAL_LEARNING_RATE,
                decay_steps=DECAY_STEPS,
                decay_rate=DECAY_RATE,
                staircase=True
            )
            
            # Compile the model with sparse categorical crossentropy loss and Adam optimizer
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                metrics=["accuracy"]
            )
            
            return model


        def plot_history(history):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')

            plt.show()


        cnn_lstm_model = build_cnn_stacked_lstm_att_model(len(label_processor.get_vocabulary()))

        filepath = "model_weights/ckpt.weights.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_accuracy", save_best_only=True, mode='max', verbose=1)

        history = cnn_lstm_model.fit(
            train_data,
            train_labels,
            # validation_split=0.2,
            validation_data=(val_data, val_labels),
            
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,

        )
        
        # cnn_lstm_model.load_weights(filepath)
        _, accuracy = cnn_lstm_model.evaluate(test_data, test_labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        plot_history(history)


        # Evaluate the model on the test set
        test_predictions = cnn_lstm_model.predict(test_data)
        test_predictions_labels = np.argmax(test_predictions, axis=1)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(test_labels, test_predictions_labels)

        # Calculate precision, recall, and F1-score
        precision = precision_score(test_labels, test_predictions_labels, average='weighted')
        recall = recall_score(test_labels, test_predictions_labels, average='weighted')
        f1 = f1_score(test_labels, test_predictions_labels, average='weighted')

        # Print the metrics
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Generate a classification report
        report = classification_report(test_labels, test_predictions_labels, target_names=label_processor.get_vocabulary())

        # Print the classification report
        print(report)

        # Plot the confusion matrix
        def plot_confusion_matrix(cm, class_names):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

        plot_confusion_matrix(conf_matrix, label_processor.get_vocabulary())

if __name__ == '__main__':
    main()
