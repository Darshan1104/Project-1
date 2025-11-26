from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys # Import sys to handle exceptions correctly

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
# Assuming you have CustomException for better error handling
# from src.exception import CustomException

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    # You likely want the landing page to be home.html if it contains the form
    # but based on your structure, index.html is the root
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Display the form page
        return render_template('home.html')
    else:
        try:
            # --- FIX APPLIED HERE ---
            # Correctly map the form inputs to the corresponding CustomData fields
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                # CORRECT ASSIGNMENT:
                reading_score=float(request.form.get('reading_score')), # <-- FIXED
                writing_score=float(request.form.get('writing_score'))  # <-- FIXED
            )
            
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("after Prediction")
            
            # The result is typically a list/array with one element, so results[0] is correct
            return render_template('home.html', results=results[0])
        
        except Exception as e:
            # This is a good place to add logging and return a useful error page
            print(f"An error occurred during prediction: {e}", file=sys.stderr)
            # You might want to render an error page or show the exception
            return render_template('home.html', results=f"Error: {e}") 
        

if __name__ == "__main__":
    # Ensure debug=True is used during development for hot-reloading and better error messages
    app.run(host="0.0.0.0", debug=True)