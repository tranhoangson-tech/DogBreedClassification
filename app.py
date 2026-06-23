import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'models/dog_breed_mobilenetv2.keras'
CLASS_NAMES = ['afghan_hound', 'airedale', 'basenji', 'beagle', 
               'bernese_mountain_dog', 'cairn', 'chow', 
               'border_terrier', 'blenheim_spaniel', 'bedlington_terrier']

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- GIAO DIỆN ---
st.title("🐶 Dog Breed Classifier")

if model is None:
    st.error(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}. Hãy chạy lại Notebook để lưu model trước!")
else:
    uploaded_file = st.file_uploader("Chọn ảnh chó...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(image, caption='Ảnh dự đoán', use_column_width=True)
        
        img_array = np.expand_dims(np.array(image), axis=0)
        
        with st.spinner('Đang phân tích...'):
            prediction = model.predict(img_array)
            score = tf.nn.softmax(prediction[0])
            breed = CLASS_NAMES[np.argmax(prediction)]
        
        st.success(f"### Kết quả: **{breed}**")
        st.write(f"Độ tự tin: **{100 * np.max(score):.2f}%**")