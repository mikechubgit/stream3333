import streamlit as st
import requests
import json
from pathlib import Path

def main():
    st.title('Bank Marketing Demo - 4 Key Features')
    st.markdown('Enter the values below for a simplified demo using 4 selected feature')

    # Load default inputs
    demo_path = Path('default_demo_input_full.json')
    with open(demo_path, 'r') as fh:
        default_input = json.load(fh)[0]
    
    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        contact_options = ['cellular', 'telephone']
        contact_value = default_input['contact']
        contact = st.selectbox(
            'Contact Type',
            options=contact_options,
            index=contact_options.index(contact_value) if contact_value in contact_options else 0

        )

        pdays = st.slider(
            'Days have passed Since Last Contact', 
             min_value=0, 
             max_value=999, 
             value=default_input['pdays'],
                          )
        

    with col2:
        month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_value = default_input['month']
        month = st.selectbox(
            'Contact Month',
            options=month_options,
            index=month_options.index(month_value) if month_value in month_options else 0
        )

        dow_options = ['mon', 'tue', 'wed', 'thu', 'fri']
        dow_value = default_input['day_of_week']
        day_of_week = st.selectbox(
            'Day of Week',
            options=dow_options,
            index=dow_options.index(dow_value) if dow_value in dow_options else 0
        )

# Merge user input into full input
    user_input = default_input.copy()
    user_input['pdays'] = pdays
    user_input['contact'] = contact
    user_input['month'] = month
    user_input['day_of_week'] = day_of_week



    print("**************  USER INPUT")

    fh = open('itaru.txt', 'w')
    fh.write(str(user_input))
    fh.close()



    print(user_input)
    print("**************  USER INPUT")   





    if st.button('predict'):
        try:
            response = requests.post('http://localhost:8000/predict/', json={'data': user_input})
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()

                # Show prediction result
                st.success(f"Prediction: {'Subscribed' if result['prediction'] else 'Not Subscribed'}")
                st.info(f"Probability of Subscribing: {result['probability']} (Threshold: {result['threshold']})")
            
            else:
                st.error('Server returned an error')
        
        except requests.exceptions.ConnectionError:
            st.error('Could not connect to FAST API backend Is it running on http://localhost:8000?')

# Allow this file to be tested on its own
if __name__ == "__main__":
    main()