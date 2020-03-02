import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
# def home():
#     return render_template('index.html')

def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)

    # con_features = np.array([int_features[0], int_features[1]]).reshape(1,-1)
    # intworth_features = np.array([int_features[0], int_features[1], int_features[2]]).reshape(1,-1)
    # apetite_features = np.array([int_features[0], int_features[2]]).reshape(1,-1)
    # sleep_features = np.array([int_features[0], int_features[1], int_features[3]]).reshape(1,-1)
    # mood_features = np.array([int_features[0], int_features[4]]).reshape(1,-1)
    con_features = np.array([11, 5]).reshape(1,-1)
    intworth_features = np.array([11, 5, 8.4]).reshape(1,-1)
    apetite_features = np.array([11, 8.4]).reshape(1,-1)
    sleep_features = np.array([11, 5, 7]).reshape(1,-1)
    mood_features = np.array([11, 241]).reshape(1,-1)

    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    con_pred = model[0].predict(con_features)
    intworth_pred = model[1].predict(intworth_features)
    apetite_pred = model[2].predict(apetite_features)
    sleep_pred = model[3].predict(sleep_features)
    mood_pred = model[4].predict(mood_features)

    score = (con_pred[0]) + (apetite_pred[0]) + (sleep_pred[0]) + (mood_pred[0]) + (intworth_pred[0] * 2) 
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Your Overall score is : {}'.format(score))


# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)