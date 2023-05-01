
import os
import flask
import markupsafe
import pickle
import pandas as pd

#This has to include loading the "backend_code" folder 
def _loadModel():
	import sys
	trainPath = os.path.abspath(os.path.join("..","..","build_model"))
	sys.path.append( os.path.abspath(os.path.join(trainPath,"backend_code")) )

	with open( os.path.join(trainPath,"model.pkl"), 'rb' ) as f:
		outModel = pickle.load(f)
	return outModel


#Initialize the app
#Using lowercase for app since it seems to be a specific flask convention
app = flask.Flask(__name__, template_folder='html_templates')
MODEL = _loadModel()


@app.route("/")
def home():
	return flask.render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
	text = list( flask.request.form.values() )[0]
	outVal = _getResponseFromInputList([text])
	outTweet = markupsafe.escape(text)
	outText = "Disaster" if outVal==1 else "Not Disaster"
	return flask.render_template("index.html", inputTweet=outTweet, predictionText=outText)


#This is the sort of "raw interface", with no attempt to return a HTML response
@app.route("/results",methods=["POST"])
def results():
	data = flask.request.get_json()
	values = data["text"]
	output = _getResponseFromInputList(values)
	return flask.jsonify( ",".join( [str(x) for x in output.tolist()] ) )

def _getResponseFromInputList(inpList):
	inpFrame = pd.DataFrame.from_dict({"text":inpList})
	output = MODEL.predict(inpFrame)
	return output


if __name__ == '__main__':
	app.run(debug=True)


