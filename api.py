import flask
from flask import Flask, render_template, request
from sklearn.externals import joblib
import featureExtractor
import regex
import predictor as phishingDetector

import sys
import logging


app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('home.html')

@app.route('/about')
def about():
    return flask.render_template('about.html')

@app.route('/predict', methods = ['POST'])
def make_prediction():
    
    if request.method=='POST':
        url = request.form['url']
        if not url:
            return render_template('home.html', label = 'Please input url')
        elif(not(regex.search(r'^(http|ftp)s?://', url))):
            return render_template('home.html', label = 'Please input full url, for exp- https://facebook.com')
        
                
        try:
            features,prediction = phishingDetector.predict(url)
            if prediction[0]==-1 :
                label = 'website is not legitimate'
            elif prediction[0]==1:
                label ='website is legitimate'

        except Exception as e:
            label = "website is not legitimate"

        
        return render_template('home.html', label=label)
        
        
if __name__ == '__main__':
    
    app.run(debug=True)
