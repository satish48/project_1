import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 Age,
                 Daily_Usage_Hours,
                 Sleep_Hours,
                 Exercise_Hours,
                 Apps_Used_Daily,
                 Time_on_Social_Media,
                 Time_on_Gaming,
                 Weekend_Usage_Hours,
                 Academic_Performance,
                 Self_Esteem,
                 Family_Communication,
                 Social_Interactions,
                 Phone_Checks_Per_Day,
                 Anxiety_Level,
                 School_Grade,
                 Gender,
                 Phone_Usage_Purpose,
                 Location,
                 Screen_Time_Before_Bed,
                 Depression_Level,
                 Parental_Control,
                 Time_on_Education
                ):
        self.Age = Age
        self.Daily_Usage_Hours = Daily_Usage_Hours
        self.Sleep_Hours = Sleep_Hours
        self.Exercise_Hours = Exercise_Hours
        self.Apps_Used_Daily = Apps_Used_Daily
        self.Time_on_Social_Media = Time_on_Social_Media
        self.Time_on_Gaming = Time_on_Gaming
        self.Weekend_Usage_Hours = Weekend_Usage_Hours
        self.Academic_Performance = Academic_Performance
        self.Self_Esteem = Self_Esteem
        self.Family_Communication = Family_Communication
        self.Social_Interactions = Social_Interactions
        self.Phone_Checks_Per_Day = Phone_Checks_Per_Day
        self.Anxiety_Level = Anxiety_Level
        self.School_Grade = School_Grade
        self.Gender = Gender
        self.Phone_Usage_Purpose = Phone_Usage_Purpose
        self.Location = Location
        self.Screen_Time_Before_Bed = Screen_Time_Before_Bed
        self.Depression_Level = Depression_Level
        self.Parental_Control = Parental_Control
        self.Time_on_Education = Time_on_Education

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'Age': [self.Age],
                'Daily_Usage_Hours': [self.Daily_Usage_Hours],
                'Sleep_Hours': [self.Sleep_Hours],
                'Exercise_Hours': [self.Exercise_Hours],
                'Apps_Used_Daily': [self.Apps_Used_Daily],
                'Time_on_Social_Media': [self.Time_on_Social_Media],
                'Time_on_Gaming': [self.Time_on_Gaming],
                'Weekend_Usage_Hours': [self.Weekend_Usage_Hours],
                'Academic_Performance': [self.Academic_Performance],
                'Self_Esteem': [self.Self_Esteem],
                'Family_Communication': [self.Family_Communication],
                'Social_Interactions': [self.Social_Interactions],
                'Phone_Checks_Per_Day': [self.Phone_Checks_Per_Day],
                'Anxiety_Level': [self.Anxiety_Level],
                'School_Grade': [self.School_Grade],
                'Gender': [self.Gender],
                'Phone_Usage_Purpose': [self.Phone_Usage_Purpose],
                'Location': [self.Location],
                'Screen_Time_Before_Bed': [self.Screen_Time_Before_Bed],
                'Depression_Level': [self.Depression_Level],
                'Parental_Control': [self.Parental_Control],
                'Time_on_Education': [self.Time_on_Education]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
