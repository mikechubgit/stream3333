

import streamlit as st
def main():
    # Title and intro about this app
    st.title('Bank Marketing Prediction App')

    st.markdown('''
                This web app predicts whether a customer will subscribe to a term deposit after a marketing campaign.

                Built with:
                - **Streamlit** (Frontend)
                - **FASTAPI**(Backend)
                - **XGBoost**(Model)

                Try the app via:
                - **Demo Mode** - Enter 4 features for a quick test
                - **Full Input Mode** - Use all 19 customer and economic variables 
                   ''')
    
    # App Architecture
    st.header('How the App works')

    st.markdown('''
                The app works using a real-time ML pipeline:

                1. Streamlit collects your input
                2. FAST API receives and process the data
                3. FAST API loads a saved *XGBoost* model
                4. A trained **XGBoost** model returns:
                - The prediction(yes/no)
                - Probability
                - Decision threshold applied
                ''')

    # Link to About the Project page
    st.markdown('''
                **Want to see how the model was built, trained and validated?**
                Click the button below to go to the detailed project breakdown. 
                ''')
    
    # Only set default once (on first load)
    if 'page' not in st.session_state:
        st.session_state.page = 'About the App'

    # Create a button to jump to About the project page
    if st.button('Go to About the Project'):
        st.session_state.page = 'About the Project'
        st.rerun()


# Allow this file to be tested on its own
if __name__ == "__main__":
    main()