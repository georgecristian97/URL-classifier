from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def getTokens(input):
	tokensBySlash = str(input.encode('utf-8')).split('/')	#get tokens after splitting by slash
	allTokens = []
	for i in tokensBySlash:
		tokens = str(i).split('-')	#get tokens after splitting by dash
		tokensByDot = []
		for j in range(0,len(tokens)):
			tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokens + tokensByDot
	allTokens = list(set(allTokens))	#remove redundant tokens
	if 'com' in allTokens:
		allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
	return allTokens

allurls = 'Data/data.csv'	#path to our all urls file
allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe

allurlsdata = np.array(allurlsdata)	#converting it into an array
random.shuffle(allurlsdata)	#shuffling

y = [d[1] for d in allurlsdata]	#all labels
corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
vectorizer = TfidfVectorizer(tokenizer=getTokens)	#get a vector for each url but use our customized tokenizer
X = vectorizer.fit_transform(corpus) #get the X vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

lgs = LogisticRegression(solver="saga",max_iter=1)	#using logistic regression
lgs.fit(X_train, y_train)
print(lgs.score(X_test, y_test)) #pring the score. It comes out to be 98%


app = Flask(__name__)
app.static_folder = 'templates'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
	userText = request.args.get('msg')
	X_predict = [str(userText)]
	X_predict = vectorizer.transform(X_predict)
	y_Predict = lgs.predict(X_predict)
	return y_Predict[0]

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000')

#X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
#X_predict = vectorizer.transform(X_predict)
#y_Predict = lgs.predict(X_predict)
#print (y_Predict) #printing predicted values
