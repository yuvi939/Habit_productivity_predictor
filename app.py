import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


with open('model.pkl','rb') as f:
    model = pickle.load(f) 

st.set_page_config(page_title="AI Productivity Predictor",layout="centered")

st.title("Habit Productivity Predictor")

# sidebar

st.sidebar.header("Enter your daily habit")

sleep = st.sidebar.slider("Sleep hours",0,12,0)
exercise = st.sidebar.slider("Exercise_hours",0,8,0)
work = st.sidebar.slider("Woerk hour",0,12,0)
breaks = st.sidebar.slider("Break taken",0,10,0)



col1, col2 = st.columns(2)
# predictions
if st.sidebar.button("Predict"):

    input_data = np.array([[sleep, exercise, work, breaks]])
    prediction = model.predict(input_data)[0]

    labels = {0: "Low", 1: "Medium", 2: "High"}
    result = labels[prediction]

    if result == "High":
        color = "#22c55e"
        emoji = "💹"
    elif result == "Medium":
        color = "#eab308"
        emoji = "⭐"
    else:
        color = "#ef4444"
        emoji = "📉"

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="title"> Productivity Level</div>
            <div class="result" style="color:{color}">
                {emoji} {result}
            </div>
        </div>
        """, unsafe_allow_html=True)

    #  Score system
    score_map = {"Low": 30, "Medium": 65, "High": 90}
    score = score_map[result]

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="title"> Productivity Score</div>
            <div class="result">{score}/100</div>
        </div>
        """, unsafe_allow_html=True)
    

    # Celebration
    if result == "High":
        st.balloons()


    score_map = {"Low": 30, "Medium": 60, "High": 90}
    st.progress(score_map[result])


# chart visualisation

st.subheader("habit overview")
habit = ['sleep','Exercise','work','breaks']
values = [sleep,exercise,work,breaks]

fig,ax = plt.subplots()
ax.bar(habit,values)
ax.set_xlabel("Habits")        
ax.set_ylabel("Values in hours")       
ax.set_title("Habit Analysis")
st.pyplot(fig)

# Feature importance

feature_names =  ['sleep','Exercise','work','breaks']
importance = model.feature_importances_

importance_data = pd.DataFrame({
    'feature':feature_names,
    'importance':importance
}).sort_values(by='importance',ascending=False)

st.subheader("Feature importance")
st.bar_chart(importance_data.set_index('feature'))


# AI suggestions

st.subheader(" AI Suggestions")

suggestions = []

# Sleep
if sleep < 6:
    suggestions.append(("warning", " Increase sleep to at least 6–8 hours for better productivity"))
elif sleep >= 8:
    suggestions.append(("success", " Great job maintaining healthy sleep!"))

# Exercise
if exercise == 1:
    suggestions.append(("warning", " Add some physical activity to boost energy"))
else:
    suggestions.append(("success", "Good job staying active!"))

# Breaks
if breaks < 2:
    suggestions.append(("info", "⏸ Taking short breaks can improve focus"))
elif breaks > 5:
    suggestions.append(("info", " Too many breaks might reduce deep work time"))

# Work hours
if work > 9:
    suggestions.append(("warning", " Too much work may reduce efficiency"))
elif work >= 6 and work <= 9:
    suggestions.append(("success", " Good productive work range"))

# Display suggestions
for level, msg in suggestions:
    if level == "warning":
        st.warning(msg)
    elif level == "success":
        st.success(msg)
    elif level == "info":
        st.info(msg)