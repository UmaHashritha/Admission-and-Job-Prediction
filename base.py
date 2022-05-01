import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model21 = pickle.load(open('model21.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/admission_predict',methods=['POST'])
def adm_predict():
    return render_template('adm_predict.html')

@app.route('/ad_predict',methods=['POST'])
def ad_predict():
    
    int_features = [float(x) for x in request.form.values()]
    for i in int_features:
        print(type(i))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]*100

    output=round(output,2)

    return render_template('adm_predict.html', prediction_text='Chance of getting admission is  {}%'.format(output))
@app.route('/job_predict',methods=['POST'])
def job_predict():
    return render_template('job_predict.html')

@app.route('/jb_predict',methods=['POST'])
def jb_predict():
    int_f = [str(x) for x in request.form.values()]
    i=int_f[:2]
    int_features =int_f[2:]
    print(int_features)
    int_features = [float(x) for x in int_features]
    if (i[0].lower()=='female'):
        int_features.extend([0,1])
    else:
        int_features.extend([1,0])
    if(i[1].lower()=='Information Technology'.lower()):
        int_features.extend([0,0,0,0,1,0])
    elif(i[1].lower()=='Computer Science'.lower()):
        int_features.extend([0,1,0,0,0,0])
    elif(i[1].lower()=='Electronics And Communication'.lower()):
        int_features.extend([0,0,0,1,0,0])
    elif(i[1].lower()=='Electrical'.lower()):
        int_features.extend([0,0,1,0,0,0])
    elif(i[1].lower()=='Mechanical'.lower()):
        int_features.extend([0,0,0,0,0,1])
    elif(i[1].lower()=='Civil'.lower()):
        int_features.extend([1,0,0,0,0,0])
    
    final_features = [np.array(int_features)]
    prediction = model21.predict(final_features)
    o=prediction[0]
    if(o==0):
        output="Your chances of getting placed are almost low to none"
    else:
        output="Your chances of getting placed are almost quite sure"



    return render_template('job_predict.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True,port=3001)
    