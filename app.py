import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
score = 0

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# con_list = []
# intworth_list = []
# apetite_list = []
# sleep_list = []
# mood_list = []
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    # con_list.append(int_features[0])
    # con_list.append(int_features[1])

    # intworth_list.append(int_features[0])
    # intworth_list.append(int_features[1])
    # intworth_list.append(int_features[2])

    # apetite_list.append(int_features[0])
    # apetite_list.append(int_features[2])

    # sleep_list.append(int_features[0])
    # sleep_list.append(int_features[1])
    # sleep_list.append(int_features[3])

    # mood_list.append(int_features[0])
    # mood_list.append(int_features[4])

    con_features = np.array([int_features[0], int_features[1]]).reshape(1,-1)
    intworth_features = np.array([int_features[0], int_features[1], int_features[2]]).reshape(1,-1)
    apetite_features = np.array([int_features[0], int_features[2]]).reshape(1,-1)
    sleep_features = np.array([int_features[0], int_features[1], int_features[3]]).reshape(1,-1)
    mood_features = np.array([int_features[0], int_features[4]]).reshape(1,-1)

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


if __name__ == "__main__":
    app.run(debug=True)