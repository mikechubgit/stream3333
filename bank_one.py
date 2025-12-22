import streamlit as st
from demo_4_features import main as page_demo_4
from full_input_demo import main as page_full
from about_app import main as page_about_app
from about_project import main as page_about_project
from contact import main as page_contact

# Get the currently selected page from session state (default to About the APP)
default_page = st.session_state.get('page', 'About the App')

# Sidebar page selector
page_names = ['Demo (4 Features)', 'Full Input', 'About the App', 'About the Project', 'Contact']
st.sidebar.title('Navigation')
page = st.sidebar.radio('Choose a page', page_names, index=page_names.index(default_page))
st.session_state.page = page

# Save the selected page into session state
st.session_state.page = page

# Run selected page
if page == 'Demo (4 Features)':
    page_demo_4()

elif page == 'Full Input':
    page_full()

elif page == 'About the App':
    page_about_app()

elif page == 'About the Project':
    page_about_project()

else:
    page_contact()


