import streamlit as st


st.title('My Title')

st.subheader('this is just sub 1')

st.header('header One')


st.caption('my caption')

st.code('s = 10')


st.write('message one')

st.checkbox('yep')

st.button('here')

st.radio('pick one',['left', 'right'])

box1 = st.selectbox('pick one',['left', 'right'])
print(box1)


st.multiselect('pick one',['left', 'middle', 'right']) 
  

st.select_slider('pick one',['left', 'middle', 'right'])

