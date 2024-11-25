import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model("cat_dog_cnn_model.h5")

# Define class labels
class_labels = ["Cat", "Dog"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Streamlit App
st.title("Cat vs Dog Image Classification")
st.write("Upload an image to classify it as a Cat or Dog.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    image = Image.open(uploaded_file)
    input_data = preprocess_image(image)
    
    # Make predictions
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    # Display results
    st.write(f"**Prediction**: {class_labels[predicted_class]}")
    st.write(f"**Confidence**: {confidence:.2f}")
    
    # Optionally display performance metrics
    if st.button("Show Model Performance"):
        # Example metrics (replace with your own if available)
        y_true = [0, 1, 0, 1, 1]  # Example true labels
        y_pred = [0, 1, 1, 0, 1]  # Example predicted labels
        
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        st.json(report)
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, class_labels)
