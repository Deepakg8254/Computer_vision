import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        self.model = None
        
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for classification
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                return None
                
            # Convert to RGB (OpenCV uses BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize image
            img = cv2.resize(img, self.image_size)
            # Normalize pixel values
            img = img / 255.0
            return img
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def load_data_from_directory(self, directory):
        """
        Load images from a specific directory structure
        """
        images = []
        labels = []
        
        # Process cat images
        cat_dir = os.path.join(directory, 'cats')
        for img_name in os.listdir(cat_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cat_dir, img_name)
                img = self.preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # 0 for cats
                
        # Process dog images
        dog_dir = os.path.join(directory, 'dogs')
        for img_name in os.listdir(dog_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dog_dir, img_name)
                img = self.preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(1)  # 1 for dogs
                
        return np.array(images), np.array(labels)

    def prepare_data(self, archive_path):
        """
        Prepare dataset from archive directory with train and test subdirectories
        """
        if not os.path.exists(archive_path):
            raise ValueError(f"Archive path {archive_path} does not exist")
            
        # Check directory structure
        required_dirs = ['train/cats', 'train/dogs', 'test/cats', 'test/dogs']
        for dir_path in required_dirs:
            full_path = os.path.join(archive_path, dir_path)
            if not os.path.exists(full_path):
                raise ValueError(f"Required directory {full_path} does not exist")
        # Load training data
        train_dir = os.path.join(archive_path, 'train')
        print("Loading training data...")
        X_train, y_train = self.load_data_from_directory(train_dir)
        
        # Load test data
        test_dir = os.path.join(archive_path, 'test')
        print("Loading test data...")
        X_test, y_test = self.load_data_from_directory(test_dir)
        
        return X_train, y_train, X_test, y_test

    def build_model(self):
        """
        Build and compile the CNN model
        """
        self.model = Sequential([
            # First Convolutional Layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Layer
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Layer
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flatten the output and add dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Train the model with data augmentation
        """
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping]
        )
        
        return history

    def plot_training_history(self, history):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy

    def predict(self, image_path):
        """
        Make prediction on a single image
        """
        # Preprocess the image
        img = self.preprocess_image(image_path)
        if img is None:
            return None, None
            
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        
        # Convert prediction to class label
        class_label = 'Dog' if prediction > 0.5 else 'Cat'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return class_label, confidence

def main():
    # Initialize classifier
    classifier = ImageClassifier()
    
    # Set the path to your archive directory
    archive_path = 'archive'
    
    # Prepare data
    print("Loading and preparing data...")
    X_train, y_train, X_test, y_test = classifier.prepare_data(archive_path)
    
    # Split training data to create validation set
    validation_split = 0.2
    split_idx = int(len(X_train) * (1 - validation_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build and train model
    print("\nBuilding and training model...")
    classifier.build_model()
    classifier.model.summary()
    
    history = classifier.train_model(X_train, y_train, X_val, y_val)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = classifier.evaluate_model(X_test, y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Example predictions
    print("\nMaking example predictions...")
    test_images = [
        os.path.join(archive_path, 'test', 'cats', os.listdir(os.path.join(archive_path, 'test', 'cats'))[0]),
        os.path.join(archive_path, 'test', 'dogs', os.listdir(os.path.join(archive_path, 'test', 'dogs'))[0])
    ]
    
    for image_path in test_images:
        class_label, confidence = classifier.predict(image_path)
        if class_label and confidence:
            print(f"\nImage: {image_path}")
            print(f"Prediction: {class_label} (Confidence: {confidence*100:.2f}%)")

if __name__ == "__main__":
    main()