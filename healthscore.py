import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('healthdata3.csv')

# Define age bins and group names
bins = [0, 20, 30, 40, 45, float('inf')]
group_names = ['1-20', '21-30', '31-40', '41-45', '45+']

# Create a new column for age groups
data['Age_Group'] = pd.cut(data['age'], bins, labels=group_names)

# Group by age range and compute the mean Health_score
mean_scores_by_age = data.groupby('Age_Group')['Health_score'].mean().reset_index()

# Select relevant features for health score prediction
features = ['age', 'weight', 'height', 'exercise_hours', 'sleep_hours', 'fruits', 'vegetables', 'junk_food']

# Splitting data
X = data[features]
y = data['Health_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Health Score and Recommendation")
st.write("Enter your information to predict your health score and receive recommendations.")
with st.sidebar:
    st.title("Mean health score")
    for index, row in mean_scores_by_age.iterrows():
        st.write(f"Age Group {row['Age_Group']}: {row['Health_score']:.2f}")

user_age = st.number_input("Enter your age:")
user_weight = st.number_input("Enter your weight (in kg):")
user_height = st.number_input("Enter your height (in meters):")
user_exercise_hours = st.number_input("Enter your exercise hours:")
user_sleep_hours = st.number_input("Enter your sleep hours:")
user_fruits = st.number_input("Enter your daily fruit intake (servings):")
user_vegetables = st.number_input("Enter your daily vegetable intake (servings):")
user_junk_food = st.number_input("Enter your daily junk food intake (servings):")

individual_data = pd.DataFrame(
    [[user_age, user_weight, user_height, user_exercise_hours, user_sleep_hours, user_fruits, user_vegetables, user_junk_food]],
    columns=features
)
health_score_prediction = model.predict(individual_data)

# Adjust the health score if it is above 100
if health_score_prediction[0] > 100:
    health_score_prediction[0] = 100

# Recommendation logic based on health score prediction
if health_score_prediction >= 80:
    recommendation = "Congratulations! Your health score is excellent. Keep up the good work with your healthy lifestyle."
elif 60 <= health_score_prediction < 80:
    exercise_hours = individual_data['exercise_hours'].values[0]
    sleep_hours = individual_data['sleep_hours'].values[0]
    fruits = individual_data['fruits'].values[0]
    vegetables = individual_data['vegetables'].values[0]
    junk_food = individual_data['junk_food'].values[0]

    exercise_recommendation = ""
    sleep_recommendation = ""
    food_recommendation = ""

    if exercise_hours < 3:
        exercise_recommendation = "Increase your daily exercise duration to at least 30 minutes for better health."
    if sleep_hours < 7:
        sleep_recommendation = "Aim for 7-9 hours of sleep each night to support your overall well-being."
    if fruits < 2:
        food_recommendation += "Increase your daily fruit intake. "
    if vegetables < 3:
        food_recommendation += "Increase your daily vegetable intake. "
    if junk_food > 1:
        food_recommendation += "Reduce your daily junk food intake."

    recommendation = f"Your health score is in a good range. {exercise_recommendation} {sleep_recommendation} {food_recommendation}"
elif 40 <= health_score_prediction < 60:
    exercise_hours = individual_data['exercise_hours'].values[0]
    sleep_hours = individual_data['sleep_hours'].values[0]
    fruits = individual_data['fruits'].values[0]
    vegetables = individual_data['vegetables'].values[0]
    junk_food = individual_data['junk_food'].values[0]

    exercise_recommendation = ""
    sleep_recommendation = ""
    food_recommendation = ""

    if exercise_hours < 3:
        exercise_recommendation = "Increasing your exercise duration can have positive impacts on your health."
    if sleep_hours < 7:
        sleep_recommendation = "Prioritize getting sufficient sleep as it contributes to your overall health."
    if fruits < 2:
        food_recommendation += "Increase your daily fruit intake. "
    if vegetables < 3:
        food_recommendation += "Increase your daily vegetable intake. "
    if junk_food > 1:
        food_recommendation += "Reduce your daily junk food intake."

    recommendation = f"Your health score suggests there's room for improvement. {exercise_recommendation} {sleep_recommendation} {food_recommendation} Focus on balanced nutrition and staying hydrated."
else:
    recommendation = "Your health score indicates potential health concerns. It's advisable to consult a healthcare professional for personalized guidance."

if st.button("Submit"):
    st.write("Predicted Health Score:", health_score_prediction[0])
    st.title("Recommendation")
    
    # Find the user's age group and display the mean health score
    user_age_group = pd.cut(pd.Series([user_age]), bins, labels=group_names)[0]
    mean_score_for_user_age_group = mean_scores_by_age[mean_scores_by_age['Age_Group'] == user_age_group]['Health_score'].values[0]
    st.write(f"Mean Health Score for Your Age Group ({user_age_group}): {mean_score_for_user_age_group:.2f}")
    
    # Calculate and display the percentage difference
    percentage_better = ((mean_score_for_user_age_group - health_score_prediction[0]) / mean_score_for_user_age_group) * 100
    st.write(f"You are better than others in your age group by: {percentage_better:.2f}%")

    st.write(recommendation)
