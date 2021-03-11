import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
model1 = pickle.load(open('model.pkl', 'rb'))
ALLOWED_HOSTS = ['*']

@app.route('/')
def home():
    return render_template('appui.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)

    output = prediction[0]

    #return render_template('appui.html', prediction_text='Predicted Class:  {}'.format(output))
    if output == 1:
      return render_template('appui.html', prediction_text='Danger! you have to take care of yourself. Tumor is Malignant{}')
    else:
     return render_template('appui.html', prediction_text=' Great, you need not to worry. Tumor is Benign {}')


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model1.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
