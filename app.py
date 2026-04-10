import streamlit as st

st.set_page_config(
    page_title="AF Risk DSS",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/Home.py",                   title="Home",             icon="🏠", default=True),
    st.Page("pages/1_Risk_Assessment.py",       title="Risk Assessment",  icon="🔍"),
    st.Page("pages/2_EDA.py",                   title="EDA",              icon="📊"),
    st.Page("pages/3_Model_Performance.py",     title="Model Performance",icon="📈"),
    st.Page("pages/4_What_If.py",               title="What-If Analysis", icon="🔬"),
])
pg.run()
