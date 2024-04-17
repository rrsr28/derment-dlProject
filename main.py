import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/NewModel2.h5")

# Define class names
class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

def predict_disease(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(450, 450), color_mode='rgb')
    array = tf.keras.utils.img_to_array(img) / 255.0

    img_array = np.expand_dims(array, axis=0)
    preds = model.predict(img_array)

    formatted_predictions = [f'{value:.2f}' for value in preds[0]]

    top_prob_index = np.argmax(formatted_predictions)
    top_prob = round(float(formatted_predictions[top_prob_index].replace(",", ".")) * 100, 2)

    # Prepare probabilities for all classes
    all_probs = {class_names[i]: float(prob) * 100 for i, prob in enumerate(formatted_predictions)}

    return img, class_names[top_prob_index], top_prob, all_probs

# Streamlit app
def main():
    st.title("Skin Disease Classification")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Store the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_img:
            temp_img.write(uploaded_file.getvalue())

        # Display the uploaded image in the first tab
        img = Image.open(temp_img.name)

        tab1, tab2, tab3 = st.tabs(["Uploaded Image", "Prediction", "Probabilities"])
        with tab1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        # Make prediction in the second tab
        with tab2:
            _, disease_class, probability, _ = predict_disease(model, temp_img.name)
            st.success(f"Predicted class: {disease_class}")
            st.success(f"Probability: {probability}%")

        # Show probabilities of all classes in the third tab
        with tab3:
            _, _, _, all_probs = predict_disease(model, temp_img.name)
            for class_name, prob in all_probs.items():
                st.write(f"- {class_name}: {prob}%")

if __name__ == "__main__":
    main()
