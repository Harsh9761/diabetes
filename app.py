import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# Load dataset
try:
    df = pd.read_csv(r"./diabetes.csv")
except FileNotFoundError:
    st.error("Dataset not found! Please ensure the file `diabetes.csv` is in the correct path.")
    st.stop()

# Add custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #eef3f8;
    font-family: 'Helvetica', sans-serif;
}

.card {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    padding: 15px 25px;
    margin-bottom: 20px;
}

.button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    text-align: center;
}

.button:hover {
    background-color: #45a049;
}

h1, h2, h3 {
    color: #34495E;
}

.success {
    color: #28a745;
    font-weight: bold;
}

.danger {
    color: #dc3545;
    font-weight: bold;
}

footer {
    margin-top: 50px;
    text-align: center;
    font-size: 0.9em;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header'>🌟 Diabetes Risk Checker</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze your health data to understand your diabetes risk.</p>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title("🩺 Enter Your Health Information")
st.sidebar.markdown("<p>Use the sliders below to provide details:</p>", unsafe_allow_html=True)

# Function to collect user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies (Number of times pregnant)', 0, 17, 3, help="Number of times you've been pregnant.")
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', 0, 200, 120, help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test.")
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 70, help="Diastolic blood pressure.")
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20, help="Thickness of the triceps skin fold.")
    insulin = st.sidebar.slider('Insulin Level (IU/mL)', 0, 846, 79, help="2-hour serum insulin concentration.")
    bmi = st.sidebar.slider('BMI (Body Mass Index)', 0.0, 67.0, 20.0, 0.1, help="Body mass index (weight in kg/(height in m)^2).")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47, 0.01, help="A function which scores likelihood of diabetes based on family history.")
    age = st.sidebar.slider('Age (years)', 21, 88, 33, help="Your age in years.")

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
st.markdown("<div class='card'><h3>Your Health Data</h3></div>", unsafe_allow_html=True)
st.write(user_data)

# Train-test split
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# Prediction and button functionality
if st.button('🔮 Check My Risk', key="predict"):
    with st.spinner('Analyzing your data...'):
        time.sleep(1)  # Simulate model processing
    prediction = rf.predict(user_data)[0]
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>🔍 Prediction Result</h2>", unsafe_allow_html=True)
    if prediction == 0:
        st.markdown("<p class='success'>✅ Low Risk: You are not diabetic.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='danger'>⚠️ High Risk: You are at risk of diabetes!</p>", unsafe_allow_html=True)
    
    st.markdown(f"<p><b>Model Accuracy:</b> {accuracy:.2f}%</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3>👈 Enter your data and click 'Check My Risk'</h3>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer>
        Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </footer>
""", unsafe_allow_html=True)
