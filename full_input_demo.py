import streamlit as st
import json
import requests
from pathlib import Path
import pandas as pd

def main():
    st.title('Full Feature input')
    st.markdown('Fill in all features below to make a prediction using the full model.')
    
    # Load default feature values from JSON
    demo_path = Path('default_demo_input_full.json')
    with open(demo_path, 'r') as fh:
        default_input = json.load(fh)[0]
    
    # Load the column names and their unique inputs form the categorical features
    cat_path = Path('full_demo_cat.json')
    with open(cat_path, 'r') as fh:
        cat_config = json.load(fh)

    # Load the column names and their input ranges from the numeric features
    num_path = Path('full_demo_num_dic.json')
    with open(num_path, 'r') as fh:
        num_config = json.load(fh)

    # Create an empty dictionary to store user inputs
    user_input = {}

    # Create a list of feature names so that they can be used as keys in the dictionary
    keys = list(default_input.keys())

    # Split 19 inputs into 6 columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    columns = [col1, col2, col3, col4, col5, col6]
    
    # Use enumerate to split the keys into 6 columns
    for i, key in enumerate(keys):
        col = columns[i % 6]
        with col:
            value = default_input[key]
            
            # Use drop down menus for categorical variables
            if key in cat_config:
                options = cat_config[key]
                user_input[key] = st.selectbox(key, options=options, index= options.index(value) if value in options else 0)
            
            # Use sliders for numeric input
            # Use float sliders for economic indicators for precision and int sliders for other numeric features
            elif key in num_config:
                if key in ['cons.conf.idx', 'cons.price.idx', 'emp.var.rate','euribor3m', 'nr.employed']:
                    user_input[key] = st.slider(key, min_value = float(num_config[key][0]), max_value= float(num_config[key][1]), value=float(value))

                else:
                    user_input[key] = st.slider(key, min_value = int(num_config[key][0]), max_value= int(num_config[key][1]), value=int(value))

    # Prediction button
    if st.button('Predict'):
        try:
        
            print("**************  USER INPUT")
            
            fh = open('itaru2.txt', 'w')
            fh.write(json.dumps(user_input))
            fh.close()
            
            
            
            print(user_input)
            print("**************  USER INPUT")            
            
            
            # response = requests.post('http://localhost:8000/predict/', json={'data': user_input})
            
           
            
            response = requests.post('https://web-production-24857.up.railway.app/predict/', json={'data': user_input})
                        
            
            # Check if the request was successful
            print('====================', response.status_code)
            
            if response.status_code == 200:
                result = response.json()

                # Show prediction result
                st.success(f"Prediction: {'Subscribed' if result['prediction'] else 'Not Subscribed'}")
                st.info(f"Probability of Subscribing: {result['probability']} (Threshold:{result['threshold']})")
            
            else:
                st.error(f'Server returned an error {response.status_code}')

        except requests.exceptions.ConnectionError:
            st.error('Could not connect to FAST API backend Is it running on http://localhost:8000?')


# Allow this file to be tested on its own
if __name__ == "__main__":
    main()