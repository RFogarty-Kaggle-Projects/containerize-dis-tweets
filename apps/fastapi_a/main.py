

import fastapi
import fastapi.templating
import markupsafe
import os
import pandas as pd
import pickle
import pydantic
import typing
from typing_extensions import Annotated

#Define POST request classes
class PredInput(pydantic.BaseModel):
	text: typing.List


#Import model
def _loadModel():
	import sys
	trainPath = os.path.abspath(os.path.join("..","..","build_model"))
	sys.path.append( os.path.abspath(os.path.join(trainPath,"backend_code")) )

	with open( os.path.join(trainPath,"model.pkl"), 'rb' ) as f:
		outModel = pickle.load(f)
	return outModel


#
app = fastapi.FastAPI()
templates = fastapi.templating.Jinja2Templates("html_templates")

#
MODEL = _loadModel()


@app.get("/")
async def home(request: fastapi.Request):
	return templates.TemplateResponse("index.html", {"request":request})

@app.post("/predict")
async def predict(request: fastapi.Request, tweetString: Annotated[str, fastapi.Form()]):
	inpTweet = markupsafe.escape(tweetString)
	modelRes = MODEL.predict( pd.DataFrame.from_dict({"text":[inpTweet]}) )[0] 
	if modelRes == 1:
		predText = "Disaster"
	else:
		predText = "Not Disaster"

	outDict = {"request":request, "inputTweet":inpTweet, "predictionText":predText}
	return templates.TemplateResponse("index.html", outDict)

@app.post("/results")
async def results(request: fastapi.Request, inpText: PredInput):
	modelOutput = MODEL.predict( pd.DataFrame.from_dict({"text":inpText.text}) )
	actOutput =  ",".join( [str(x) for x in modelOutput.tolist()] )
	encodedOutput = fastapi.encoders.jsonable_encoder(actOutput)
	return fastapi.responses.JSONResponse(content=encodedOutput)




