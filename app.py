import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CẤU HÌNH ---
MODEL_PATH = 'models/dog_breed_mobilenetv2.keras'
CLASS_NAMES = ['afghan_hound', 'airedale', 'basenji', 'beagle', 
                'bedlington_terrier', 'bernese_mountain_dog', 'blenheim_spaniel', 
                'border_terrier', 'cairn', 'chow']

# --- LOAD MODEL (BÍ KÍP Ở ĐÂY) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    # Chúng ta truyền 'custom_objects' để Keras tự tìm thấy hàm preprocess_input mà nó đang cần
    return tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'preprocess_input': preprocess_input}
    )

model = load_model()

# --- GIAO DIỆN (Giữ nguyên phần dưới) ---
st.title("🐶 Dog Breed Classifier")

if model is None:
    st.error(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}.")
else:
    uploaded_file = st.file_uploader("Chọn ảnh chó...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(image, caption='Ảnh dự đoán', use_column_width=True)
        
        img_array = np.expand_dims(np.array(image), axis=0)
        
        with st.spinner('Đang phân tích...'):
            # Model đã có Lambda layer bên trong, nó sẽ tự gọi preprocess_input 
            # mà ta đã khai báo ở trên nên code này sẽ chạy ngon lành!
            prediction = model.predict(img_array)
            breed = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])
        
        st.success(f"### Kết quả: **{breed.replace('_', ' ').title()}**")
        st.write(f"Độ tự tin: **{100 * confidence:.2f}%**")