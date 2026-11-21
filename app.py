from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_marks = float(request.form["assignment_marks"])
    input_data = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks
    ]])
    prediction = model.predict(input_data)
    result = prediction[0]
    return render_template(
        "index.html",
        prediction=result
    )
if __name__ == "__main__":
    app.run()