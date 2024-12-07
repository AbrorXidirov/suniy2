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
make_model = st.selectbox('Make and Model', ['Audi A3', 'BMW X5', 'Mercedes Benz A-Class', 'Toyota Corolla'])
gearing_type = st.selectbox('Gearing Type', ['Automatic', 'Manual'])

# Kategorik o'zgaruvchilarni one-hot encoding yordamida o'zgartirish
make_model_audi = 1 if make_model == 'Audi A3' else 0
make_model_bmw = 1 if make_model == 'BMW X5' else 0
make_model_mercedes = 1 if make_model == 'Mercedes Benz A-Class' else 0
make_model_toyota = 1 if make_model == 'Toyota Corolla' else 0

gearing_type_automatic = 1 if gearing_type == 'Automatic' else 0
gearing_type_manual = 1 if gearing_type == 'Manual' else 0

# Kirish ma'lumotlarini DataFrame formatida tayyorlash
input_data = pd.DataFrame({
    'hp_kW': [hp_kW],
    'age': [age],
    'km': [km],
    'make_model_Audi A3': [make_model_audi],
    'make_model_BMW X5': [make_model_bmw],
    'make_model_Mercedes Benz A-Class': [make_model_mercedes],
    'make_model_Toyota Corolla': [make_model_toyota],
    'gearing_type_Automatic': [gearing_type_automatic],
    'gearing_type_Manual': [gearing_type_manual]
})

# `input_data` ni model uchun kerakli formatga moslashtirish
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Model yordamida narxni bashorat qilish
if st.button('Predict Price'):
    predicted_price = model.predict(input_data)
    
    # Natijani chiqarish
    st.write(f'The predicted price for the car is: ${predicted_price[0]:,.2f}')
