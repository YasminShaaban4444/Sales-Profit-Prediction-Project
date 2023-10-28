import streamlit as st
import pandas as pd
import numpy as np
from models.dummies import *
import joblib
import logging

model=joblib.load('./models/linear_reg_model.h5')
scaler=joblib.load('./models/scaler.h5')

#df=pd.read_csv('data/Life_Expectancy_Data.csv')

st.title('Sales Profit Prediction Project')
#st.info('Just buiding a model')

# col1,col2,col3=st.columns(3)
# col1.metric('temp','234')
# col2.metric('hum','2345')
# col3.metric('weather','clear')

# temp=st.number_input('Enter Temprature: ')
# humidity=st.number_input('Enter humidity: ')
year=st.number_input('Enter year: ',min_value=2015)
cost=st.number_input('Enter cost: ',min_value=5.0)
unit_cost=st.number_input('Enter unit cost: ',min_value=5.0)
unit_price=st.number_input('Enter unit price: ',min_value=5.0)
customer_age=st.number_input('Enter customer age: ',min_value=10)
hour=st.slider('Hour?',0,24,16)
country=st.selectbox('Country? ',['United States','United Kingdom','France','Germany']) 
country_selected=country_dummies[country]
product_category=st.selectbox('Product Category? ',['Accessories','Bikes','Clothing']) 
product_category_selected=product_category_dummies[product_category]

if(product_category == 'Accessories'):
   sub_category=st.selectbox('Sub category? ',['Tires and Tubes','Bottles and Cages','Helmets','Fenders','Cleaners','Hydration Packs','Bike Stands','Bike Racks'])
elif(product_category == 'Bikes'):
   sub_category=st.selectbox('Sub category? ',['Road Bikes','Mountain Bikes','Touring Bikes'])
elif(product_category == 'Clothing'):
   sub_category=st.selectbox('Sub category? ',['Jerseys','Caps','Shorts','Gloves','Socks','Vests'])

sub_category_selected=sub_category_dummies[sub_category]

gender=st.selectbox('Gender? ',['Male','Female']) 
gender_selected=customer_gender_dummies[gender]

data=[year,customer_age,unit_cost,unit_price,cost,hour]
data.extend(country_selected)
data.extend(product_category_selected)
data.extend(sub_category_selected)
data.extend(gender_selected)

#st.write(data)
data_scaled=scaler.transform([data])

try:
    st.header("Expected Profit is: ")
    result=model.predict(np.array([data]))
    st.write(result)
    #st.info("Expected profit is : ",result)
except Exception as e:
    st.write("Something occurs when showing results")
    logging.error("Exception occurs",exc_info=True)
   



