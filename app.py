from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from sklearn.preprocessing import StandardScaler
from src.utils import get_addiction_label

application = Flask(__name__)

app = application

@app.route('/', methods=['GET'])
def home_page():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Age=int(request.form.get('Age')),
            Daily_Usage_Hours=float(request.form.get('Daily_Usage_Hours')),
            Sleep_Hours=float(request.form.get('Sleep_Hours')),
            Exercise_Hours=float(request.form.get('Exercise_Hours')),
            Apps_Used_Daily=int(request.form.get('Apps_Used_Daily')),
            Time_on_Social_Media=float(request.form.get('Time_on_Social_Media')),
            Time_on_Gaming=float(request.form.get('Time_on_Gaming')),
            Weekend_Usage_Hours=float(request.form.get('Weekend_Usage_Hours')),
            Academic_Performance=float(request.form.get('Academic_Performance')),
            Self_Esteem=float(request.form.get('Self_Esteem')),
            Family_Communication=float(request.form.get('Family_Communication')),
            Social_Interactions=float(request.form.get('Social_Interactions')),
            Phone_Checks_Per_Day=int(request.form.get('Phone_Checks_Per_Day')),
            Anxiety_Level=float(request.form.get('Anxiety_Level')),
            School_Grade=request.form.get('School_Grade'),
            Gender=request.form.get('Gender'),
            Phone_Usage_Purpose=request.form.get('Phone_Usage_Purpose'),
            Location=request.form.get('Location'),
            Screen_Time_Before_Bed=float(request.form.get('Screen_Time_Before_Bed')),
            Depression_Level=float(request.form.get('Depression_Level')),
            Parental_Control=float(request.form.get('Parental_Control')),
            Time_on_Education=float(request.form.get('Time_on_Education'))
        )

        prediction_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(prediction_df)

        output = round(prediction[0], 2)
        label = get_addiction_label(output)

        return render_template(
            'home.html',
            prediction_text=f"Predicted Addiction Level: {output} / 10 ({label})"
        )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)