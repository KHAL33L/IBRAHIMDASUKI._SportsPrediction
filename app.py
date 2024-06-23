import streamlit as st
import joblib
import pandas as pd



# Load model
file_path = "fifa_rating_predictor.pkl"
try:
    model = joblib.load(file_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Setting common features
common_features = [
    'age', 'skill_ball_control', 'skill_long_passing', 'power_shot_power', 
    'potential', 'attacking_short_passing', 'mentality_vision', 
    'movement_reactions', 'mentality_composure'
]

# Preprocessing function
def preprocess(data):
    input_df = pd.DataFrame([data])
    for feature in common_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[common_features]
    return input_df

# Prediction function
def predict(data):
    processed_input = preprocess(data)
    prediction = model.predict(processed_input)
    return prediction[0]

st.title('FIFA Player Rating Predictor')

# Input fields for the features
age = st.number_input('Age', min_value=15, max_value=50)
skill_ball_control = st.slider('Ball Control', 0, 100)
skill_long_passing = st.slider('Long Passing', 0, 100)
power_shot_power = st.slider('Power Shot', 0, 100)
potential = st.slider('Potential', 0, 100)
attacking_short_passing = st.slider('Short Passing', 0, 100)
mentality_vision = st.slider('Mentality Vision', 0, 100)
movement_reactions = st.slider('Movement Reactions', 0, 100)
mentality_composure = st.slider('Mentality Composure', 0, 100)

# Button for prediction
if st.button('Predict Rating'):
    data = {
        'age': age,
        'skill_ball_control': skill_ball_control,
        'skill_long_passing': skill_long_passing,
        'power_shot_power': power_shot_power,
        'potential': potential,
        'attacking_short_passing': attacking_short_passing,
        'mentality_vision': mentality_vision,
        'movement_reactions': movement_reactions,
        'mentality_composure': mentality_composure
    }
    
    # Prediction
    try:
        prediction = predict(data)
        st.success(f'Player rating prediction: {prediction:.2f}')
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")
