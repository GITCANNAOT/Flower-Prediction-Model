import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load model
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Iris ML Predictor",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Dark Mode Toggle
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# -----------------------------
# Background Color
# -----------------------------
if dark_mode:
    bg = "#0f172a"
else:
    bg = "linear-gradient(120deg,#89f7fe,#66a6ff)"

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown(f"""
<style>

body {{
background:{bg};
}}

.title {{
text-align:center;
font-size:70px;
font-weight:900;
color:white;
margin:0;
padding:0;
line-height:1.1;
text-shadow:0px 0px 10px rgba(255,255,255,0.6);
}}

.subtitle {{
text-align:center;
font-size:22px;
font-weight:bold;
color:white;
margin:0;
padding:0;
margin-bottom:10px;
}}

.card {{
background: rgba(255,255,255,0.2);
padding:25px;
border-radius:20px;
backdrop-filter: blur(10px);
box-shadow:0 8px 32px rgba(0,0,0,0.2);
}}

.prediction {{
font-size:28px;
font-weight:bold;
color:#00ff7f;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌸 About App")

st.sidebar.write("""
This app predicts **Iris flower species** using a trained Machine Learning model.

Dataset: Iris Dataset  
Algorithm: Scikit-Learn Model  
Framework: Streamlit
""")

# -----------------------------
# Header
# -----------------------------
st.markdown(
'<p class="title">🌸 Iris Flower Species Predictor</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Machine Learning model that predicts Iris species using flower measurements</p>',
unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# Input Section
# -----------------------------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔧 Enter Flower Measurements")

    sepal_length = st.slider("Sepal Length (cm)",4.0,8.0,5.4)
    sepal_width = st.slider("Sepal Width (cm)",2.0,4.5,3.4)
    petal_length = st.slider("Petal Length (cm)",1.0,7.0,1.3)
    petal_width = st.slider("Petal Width (cm)",0.1,2.5,0.2)

    predict_button = st.button("🚀 Predict Species")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Section
# -----------------------------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Result")

    if predict_button:

        with st.spinner("🔍 Analyzing Flower..."):
            time.sleep(1.5)

        features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        result = encoder.inverse_transform(prediction)[0]

        st.balloons()

        st.markdown(
        f'<p class="prediction">🌼 Predicted Species: {result}</p>',
        unsafe_allow_html=True
        )

        # Flower Image
        if result == "setosa":
            st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
            width=220)

        elif result == "versicolor":
            st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            width=220)

        elif result == "virginica":
            st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
            width=220)

        # Probability Chart
        st.subheader("📈 Prediction Probability")

        species = encoder.classes_
        probs = probabilities[0]

        fig, ax = plt.subplots(figsize=(4,2.5))

        ax.bar(species, probs)
        ax.set_ylabel("Probability")

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Dataset Visualization
# -----------------------------
st.divider()
st.subheader("📊 Iris Dataset Preview")

df = pd.read_csv(
"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

st.dataframe(df.head())

# Feature Distribution Chart
st.subheader("📈 Feature Distribution")

fig2, ax2 = plt.subplots()
df["sepal_length"].hist(ax=ax2)

st.pyplot(fig2)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
"""
<center>

🌸 **Machine Learning Portfolio Project**

Built with **Streamlit, Scikit-Learn & Python**

</center>
""",
unsafe_allow_html=True
)