import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Add this import statement

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image, target_size):
    image = tf.keras.utils.load_img(image, target_size=target_size)
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

def main():
    st.header("Image Classification Model")

    model_path = "D:\Google drive\Fruits_Vegetables\Image_Recognition.keras"
    model = load_model(model_path)

    data_categories = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
                       'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
                       'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
                       'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
                       'sweetpotato', 'tomato', 'turnip', 'watermelon']

    uploaded_files = st.file_uploader("Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                st.write(f"### Image: {uploaded_file.name}")

                # Display original image
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_column_width=True)

                # Preprocess and display resized image
                resized_image = preprocess_image(uploaded_file, target_size=(180, 180))
                # st.image(resized_image[0], caption="Resized Image", use_column_width=True)

                # Make prediction
                st.set_option('deprecation.showPyplotGlobalUse', False)
                prediction = model.predict(resized_image)
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                predicted_class = data_categories[predicted_class_index]

                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                # Visualization of Prediction Probabilities
                plt.bar(data_categories, prediction[0])
                plt.xlabel('Fruit/Vegetable Class')
                plt.ylabel('Probability')
                plt.title('Prediction Probabilities')
                plt.xticks(rotation=90)  # Rotate x-axis labels

                st.pyplot()

                st.write("---")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    if st.button("Reset"):
        # Clear uploaded files and results
        uploaded_files.clear()

if __name__ == "__main__":
    main()
