import streamlit as st
import pickle

# Modelni yuklash
with open('lasso_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Kiritish maydonlari
st.title('Car Price Prediction')

# Foydalanuvchi uchun inputlar
hp_kW = st.number_input('Horsepower (kW)', min_value=0, max_value=1000, value=66)
age = st.number_input('Age (years)', min_value=0, max_value=50, value=2)
km = st.number_input('Kilometers driven', min_value=0, max_value=500000, value=17000)
make_model = st.selectbox('Make and Model', ['Audi A3', 'BMW X5', 'Mercedes Benz A-Class', 'Toyota Corolla'])
gearing_type = st.selectbox('Gearing Type', ['Automatic', 'Manual'])

# Model yordamida narxni bashorat qilish
if st.button('Predict Price'):
    # Kirish qiymatlari
    input_data = [[hp_kW, age, km, make_model, gearing_type]]

    # Modelni ishlatish
    predicted_price = model.predict(input_data)

    # Natijani chiqarish
    st.write(f'The predicted price for the car is: ${predicted_price[0]:,.2f}')
