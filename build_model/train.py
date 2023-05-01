
import os
import pickle

import numpy as np
import pandas as pd

import shared_code_path
import model_wrappers as modelWrapHelp
import standard_pipes as stdPipeHelp
import train_pipes as trainPipeHelp

# Configuration variables
TRAIN_PATH = os.path.abspath( os.path.join("train.csv") )
MODEL_PATH = "model.pkl"
RANDOM_SEED = 523423

#Model hyperparameters
C_VALUE = 1e1
MAX_DF = 1e0
MIN_DF = 1e-4

#Set random seed
np.random.seed(RANDOM_SEED)

#Prepare the training data
RAW_TRAIN = pd.read_csv(TRAIN_PATH)
cleanPipe = stdPipeHelp.loadTextPreprocPipeA()
PROC_DATA = cleanPipe.fit_transform(RAW_TRAIN)
PROC_DATA = PROC_DATA.sample(frac=1.0)

#Fit the model
trainPipe = trainPipeHelp.AddBagOfWords(vectKwargs={"max_df":MAX_DF, "min_df":MIN_DF})
FINAL_MODEL = modelWrapHelp.LogRegressionClassifier(feats=list(), featPrefix=["bow_"], trainPipe=trainPipe,
                                                    C=C_VALUE, modelKwargs={"max_iter":1000})
FINAL_MODEL.fit(PROC_DATA)

#Add a text cleaning pipeline to the final model
FINAL_PIPELINE = stdPipeHelp.loadTextPreprocPipeA(removeDuplicateTweets=False)
FINAL_PIPELINE.steps.append( ("Model",FINAL_MODEL) )

#Double check our output model has good accuracy on the train set
useTrain = RAW_TRAIN[ ["text"] ]
useTrain.head(2)
trainPred = FINAL_PIPELINE.predict(useTrain)
_deltaVals = [ abs(pred-act) for pred,act in zip(trainPred,RAW_TRAIN["target"].to_numpy()) ]
trainAcc = (len(_deltaVals)-sum(_deltaVals)) / len(_deltaVals)
print("Training accuracy = {}".format(trainAcc))

# Write our model out, including the pipeline
pickle.dump(FINAL_PIPELINE, open(MODEL_PATH,'wb'))


