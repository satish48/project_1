from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
       data = CustomData(
            Age=int(request.form.get('age')),
            Daily_Usage_Hours=float(request.form.get('daily_usage_hours')),
            Sleep_Hours=float(request.form.get('sleep_hours')),
            Exercise_Hours=float(request.form.get('exercise_hours')),
            Apps_Used_Daily=int(request.form.get('apps_used_daily')),
            Time_on_Social_Media=float(request.form.get('time_on_social_media')),
            Time_on_Gaming=float(request.form.get('time_on_gaming')),
            Weekend_Usage_Hours=float(request.form.get('weekend_usage_hours')),
            Academic_Performance=float(request.form.get('academic_performance')),
            Self_Esteem=float(request.form.get('self_esteem')),
            Family_Communication=float(request.form.get('family_communication')),
            Social_Interactions=float(request.form.get('social_interactions')),
            Phone_Checks_Per_Day=int(request.form.get('phone_checks_per_day')),
            Anxiety_Level=float(request.form.get('anxiety_level')),
            School_Grade=request.form.get('school_grade'),
            Gender=request.form.get('gender'),
            Phone_Usage_Purpose=request.form.get('phone_usage_purpose'),
            Location=request.form.get('location'),
            Screen_Time_Before_Bed=float(request.form.get('screen_time_before_bed')),
            Depression_Level=float(request.form.get('depression_level')),
            Parental_Control=float(request.form.get('parental_control')),
            Time_on_Education=float(request.form.get('time_on_education'))
        )



    prediction_df = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()

    results = predict_pipeline.predict(prediction_df)
    return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)