import config
from sklearn.externals import joblib
import  featureExtractor
#sample phishing url: http://globalsign.uk.virus-control.com/b4df29/?login_id=1817

#load the pickle file
classifier = joblib.load('rf_final.pkl')


def predict(url):
        features = featureExtractor.main(url)
        prediction = classifier.predict(features)
        
        return features,prediction
        
if __name__ == '__main__':
#input url
        print("enter url")
        url = input()

        #checking and predicting
        checkprediction = featureExtractor.main(url)
        prediction = classifier.predict(checkprediction)
        print("Malicious" if prediction[0] == -1 else "Safe")
