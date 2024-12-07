import streamlit as st
import pickle
import pandas as pd

# Modelni yuklash (GridSearchCV bilan eng yaxshi modelni olish)
with open('lasso_model.pkl', 'rb') as file:
    grid_search = pickle.load(file)

# GridSearchCV obyekti ustida eng yaxshi modelni olish
model = grid_search.best_estimator_

# Foydalanuvchidan ma'lumotlarni olish
st.title('Car Price Prediction')

# Foydalanuvchi uchun inputlar
hp_kW = st.number_input('Horsepower (kW)', min_value=0, max_value=1000, value=66)
age = st.number_input('Age (years)', min_value=0, max_value=50, value=2)
km = st.number_input('Kilometers driven', min_value=0, max_value=500000, value=17000)

# Foydalanuvchidan faqat ma'lum `make_model` tanlovini olish
make_model = st.selectbox('Make and Model', ['Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia',
                                             'Renault Clio', 'Renault Duster', 'Renault Espace'])

# Gearing type faqat tanlangan qiymat bo'lishi kerak
gearing_type = st.selectbox('Gearing Type', ['Automatic', 'Manual', 'Semi-automatic'])

# Kategorik o'zgaruvchilarni one-hot encoding yordamida o'zgartirish
make_model_audi_a1 = True if make_model == 'Audi A1' else False
make_model_audi_a3 = True if make_model == 'Audi A3' else False
make_model_opel_astra = True if make_model == 'Opel Astra' else False
make_model_opel_corsa = True if make_model == 'Opel Corsa' else False
make_model_opel_insignia = True if make_model == 'Opel Insignia' else False
make_model_renault_clio = True if make_model == 'Renault Clio' else False
make_model_renault_duster = True if make_model == 'Renault Duster' else False
make_model_renault_espace = True if make_model == 'Renault Espace' else False

gearing_type_automatic = True if gearing_type == 'Automatic' else False
gearing_type_manual = True if gearing_type == 'Manual' else False
gearing_type_semi_automatic = 1 if gearing_type == 'Semi-automatic' else False

# Kirish ma'lumotlarini DataFrame formatida tayyorlash
input_data = pd.DataFrame({
    'hp_kW': [hp_kW],
    'km': [km],
    'age': [age],
    'make_model_Audi A1': [make_model_audi_a1],
    'make_model_Audi A3': [make_model_audi_a3],
    'make_model_Opel Astra': [make_model_opel_astra],
    'make_model_Opel Corsa': [make_model_opel_corsa],
    'make_model_Opel Insignia': [make_model_opel_insignia],
    'make_model_Renault Clio': [make_model_renault_clio],
    'make_model_Renault Duster': [make_model_renault_duster],
    'make_model_Renault Espace': [make_model_renault_espace],
    'Gearing_Type_Automatic': [gearing_type_automatic],
    'Gearing_Type_Manual': [gearing_type_manual],
    'Gearing_Type_Semi-automatic': [gearing_type_semi_automatic]
})

# `input_data` ni model uchun kerakli formatga moslashtirish
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Model yordamida narxni bashorat qilish
if st.button('Predict Price'):
    predicted_price = model.predict(input_data)
    
    # Natijani chiqarish
    st.write(f'The predicted price for the car is: ${predicted_price[0]:,.2f}')
