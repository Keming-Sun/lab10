import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

st.title('California housing data(1990)by Lihong Luo')
df = pd.read_csv('housing.csv')

housing_price_filter = st.slider('Median House Price :', 0, 500001, 200000)  # min, max, default


# create a multi select
location_filter = st.sidebar.multiselect(
     'Choose the location type',
     df.ocean_proximity.unique(),  # options
     df.ocean_proximity.unique()
)  

income_filter = st.sidebar.radio(
     "Choose income level",
     ('Low','Medium','High')
)
    
# filter by price
df = df[df.median_house_value >= housing_price_filter]   

#filter by location
df = df[df.ocean_proximity.isin(location_filter)]

#filter by income
if income_filter == 'Low':
    df = df[df.median_income<=2.5]
         
if income_filter == 'Medium':   
    df = df[(df.median_income > 2.5) & (df.median_income < 4.5 )]

if income_filter == 'High':
    df = df[df.median_income>4.5]

st.subheader('See more filters in the sidebar:')

# show on map
st.map(df)

# show the plot
st.subheader('Historgam of the Median House Value')

fig, ax = plt.subplots()
pop_median_house_value = df.median_house_value.hist(bins=30)

st.pyplot(fig)