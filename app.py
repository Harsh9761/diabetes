import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# Load dataset
df = pd.read_csv(r"./diabetes.csv")

# Add custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .header {
            font-size: 50px;
            font-family: 'Arial Black', sans-serif;
            text-align: center;
            background: linear-gradient(to right, #ff5e5e, #ffaa00);
            -webkit-background-clip: text;
            color: transparent;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .sub-header {
            font-size: 18px;
            color: #333;
            text-align: center;
        }
        .highlight {
            font-size: 20px;
            color: #ff4b4b;
            font-weight: bold;
            text-align: center;
        }
        .card {
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 10px;
        }
        .button {
            background-color: #ff5e5e;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
        }
        .button:hover {
            background-color: #ff4040;
            color: white;
        }
        .success {
            color: #28a745;
            font-size: 24px;
            font-weight: bold;
        }
        .danger {
            color: #dc3545;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
# Beautiful heading with gradient, shadow, and stylish font
st.markdown("""
    <style>
        .main-header {
            font-size: 60px;
            font-family: 'Arial Black', sans-serif;
            text-align: center;
            background: linear-gradient(to right, #ff5e5e, #ffaa00, #00c9ff);
            -webkit-background-clip: text;
            color: transparent;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 20px;
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #333333;
            margin-bottom: 30px;
        }
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #ffaa00, #ff5e5e);
        }
    </style>
    <h1 class="main-header">🌟 Diabetes Risk Checker 🌟</h1>
    <p class="sub-header">Analyze your health data and uncover potential risks in just seconds!</p>
    <hr>
""", unsafe_allow_html=True)


# Header
st.markdown("<h1 class='header'>✨ Diabetes Risk Checker ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze your health data to understand your diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title("🩺 Enter Health Information")
st.sidebar.markdown("<p>Adjust the sliders to input your details:</p>", unsafe_allow_html=True)

# Function to collect user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', min_value=0, max_value=17, value=3, format="%d")
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, format="%d")
    bmi = st.sidebar.slider('BMI (Body Mass Index)', min_value=0.0, max_value=67.0, value=20.0, format="%.1f")
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, format="%d")
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20, format="%d")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, format="%.2f")
    insulin = st.sidebar.slider('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79, format="%d")
    age = st.sidebar.slider('Age (years)', min_value=21, max_value=88, value=33, format="%d")

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(user_data, index=[0])

user_data = get_user_input()

# Display user input
st.markdown("<div class='card'><h3 style='color: #ff5e5e;'>Your Health Data</h3></div>", unsafe_allow_html=True)
st.write(user_data)

# Train-test split
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Prediction and button functionality
if st.button('🔮 Check My Risk', key="predict"):
    st.markdown("<h3 style='text-align: center;'>🔄 Analyzing...</h3>", unsafe_allow_html=True)
    progress = st.progress(0)
    for percent in range(101):
        time.sleep(0.02)
        progress.progress(percent)

    prediction = rf.predict(user_data)[0]
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #ff5e5e;'>Prediction Result</h2>", unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<p class='success'>✅ You are not diabetic!</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='danger'>⚠️ You are at risk of diabetes!</p>", unsafe_allow_html=True)
    
    st.markdown(f"<p class='highlight'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center;'>👈 Enter your data and click 'Check My Risk'</h3>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer style="text-align: center; margin-top: 50px; color: #aaa;">
        Built with ❤️ using <a href="https://streamlit.io" style="color: #ffaa00; text-decoration: none;" target="_blank">Streamlit</a>
    </footer>
""", unsafe_allow_html=True)
