import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

# Modelni va scalerni yuklash
with open('lasso_model.pkl', 'rb') as file:
    final_model = pickle.load(file)

with open('final_scaler.pkl', 'rb') as file:
    final_scaler = pickle.load(file)

# Kiritish maydonlari
st.title('Avtomobil Narxini Bashorat Qilish')

# Foydalanuvchi uchun inputlar
hp_kW = st.number_input('Ot kuchi (kW)', min_value=0, max_value=1000, value=66)
age = st.number_input('Ishlab chiqarilganiga qancha bo'lgan (yil)', min_value=0, max_value=50, value=2)
km = st.number_input('Yurilgan masofa (km)', min_value=0, max_value=500000, value=17000)
make_model = st.selectbox('Marka va Model', ['Audi A3', 'BMW X5', 'Mercedes Benz A-Class', 'Toyota Corolla'])
gearing_type = st.selectbox('Uzatish turi', ['Avtomatik', 'Qo'lda'])

# Model yordamida narxni bashorat qilish
if st.button('Narxni bashorat qilish'):
    # Kirish qiymatlari
    my_dict = {
        "hp_kW": hp_kW,
        "age": age,
        "km": km,
        "make_model": make_model,
        "Gearing_Type": gearing_type
    }

    # DataFrame'ga o'zgartirish
    my_dict = pd.DataFrame([my_dict])

    # One-hot encoding (kategorik ustunlar uchun)
    my_dict = pd.get_dummies(my_dict)

    # X.columns bilan moslashtirish
    my_dict = my_dict.reindex(columns=['hp_kW', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
                                       'make_model_Opel Astra', 'make_model_Opel Corsa', 'make_model_Opel Insignia',
                                       'make_model_Renault Clio', 'make_model_Renault Duster', 'make_model_Renault Espace',
                                       'Gearing_Type_Automatic', 'Gearing_Type_Manual', 'Gearing_Type_Semi-automatic'], fill_value=0)

    # Ma'lumotni normallashtirish
    my_dict_scaled = final_scaler.transform(my_dict)

    # Model yordamida narxni bashorat qilish
    predicted_price = final_model.predict(my_dict_scaled)

    # Natijani chiqarish
    st.write(f'Avtomobilning bashorat qilingan narxi: ${predicted_price[0]:,.2f}')
