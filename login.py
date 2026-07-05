import streamlit as st

st.set_page_config(page_title="Deepfake Detection Login")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.title("🎥 Deepfake Detection System")

if not st.session_state.logged_in:

    username = st.text_input("Username")

    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

else:

    st.success("Welcome!")

    st.write("Login Successful")

    st.write("Now run:")

    st.code("streamlit run dashboard.py")