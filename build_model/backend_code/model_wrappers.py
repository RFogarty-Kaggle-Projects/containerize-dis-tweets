
import sklearn as sk
import sklearn.base
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm

class ScoreMixin():

	def score(self, inpX, inpY=None, sample_weight=None):
		predVals = self.predict(inpX)
		actVals = inpX[self.targCol]
		return _getF1FromPredAndAct(predVals, actVals)

def _getF1FromPredAndAct(predVals, actVals):
	actTrue = sum(actVals)
	truePos = sum( [ (act==1) & (exp==1) for exp,act in zip(predVals,actVals)  ] )
	falsePos = sum( [ (act==0) & (exp==1) for exp,act in zip(predVals, actVals)  ] )
	falseNeg = sum( [ (act==1) & (exp==0) for exp,act in zip(predVals, actVals) ] )

	recall = truePos / (truePos + falseNeg)

	if (truePos+falsePos) == 0:
		print("WARNING: TP + FP = 1; this means precision isnt particularly well defined but will be set to 1.0")
		precision = 1.0
	else:
		precision = truePos / (truePos+falsePos)

	outScore = 2*( (precision*recall) / (precision+recall)  )
	return outScore


class GenericSklearnWrapper( sk.base.BaseEstimator, ScoreMixin ):

	def __init__(self, feats, modelClass, targCol="target", featPrefix=None, trainPipe=None, modelKwargs=None):
		self.feats = feats
		self.targCol = targCol
		self.featPrefix = list() if featPrefix is None else featPrefix
		self.trainPipe = trainPipe
		self.modelKwargs = modelKwargs #needed for inherited clone method....very stupid

		_currKwargs = dict() if modelKwargs is None else modelKwargs
		self.classifier = modelClass(**_currKwargs)

	def fit(self, inpX, y=None):
		#Modify training data if required
		if self.trainPipe is not None:
			trainData = self.trainPipe.fit_transform(inpX)
		else:
			trainData = inpX

		#Fit a random forest classifier
		useTrain = self._getRelColsFromFrame(trainData)
		self.classifier.fit( useTrain.to_numpy(), trainData[self.targCol] )

		return self

	def predict(self, inpX):
		#Apply processing
		if self.trainPipe is not None:
			useX = self.trainPipe.transform(inpX)
		else:
			useX = inpX.copy()

		#
		useX = self._getRelColsFromFrame(useX)
		return self.classifier.predict(useX.to_numpy())

	def _getRelColsFromFrame(self, inpX):
		featNames = self._getInpFeatNamesFromColNames(inpX.columns)
		return inpX[featNames]

	def _getInpFeatNamesFromColNames(self, allCols):
		#Get column names based on the prefixes
		prefixCols = list()
		for prefix in self.featPrefix:
			currCols = [x for x in allCols if x.startswith(prefix) and x not in prefixCols]
			prefixCols.extend(currCols)

		#Get other feature names that DONT appear in the prefix list
		featCols = list()
		for feat in self.feats:
			#Note: all returns True for an empty list
			if all([not feat.startswith(prefix) for prefix in self.featPrefix]):
				featCols.append(feat)

		return featCols + prefixCols



class LogRegressionClassifier( GenericSklearnWrapper ):

	def __init__(self, feats, targCol="target", featPrefix=None, trainPipe=None, modelKwargs=None, C=None,
	             l1_ratio=None, penalty=None):
		modelClass = sk.linear_model.LogisticRegression
		_currKwargs = {"targCol":targCol, "featPrefix":featPrefix, "trainPipe":trainPipe, "modelKwargs":modelKwargs}
		super().__init__(feats, modelClass, **_currKwargs)

		#Deal with model kwargs; doing it this way lets me use sklearn grid search
		if C is not None:
			self.C = C

		if l1_ratio is not None:
			self.l1_ratio = l1_ratio

		if penalty is not None:
			self.penalty = penalty

	@property
	def C(self):
		return self.classifier.C 

	@C.setter
	def C(self, value):
		self.classifier.C = value

	@property
	def l1_ratio(self):
		return self.classifier.l1_ratio

	@l1_ratio.setter
	def l1_ratio(self, value):
		self.classifier.l1_ratio = value

	@property
	def penalty(self):
		return self.classifier.penalty

	@penalty.setter
	def penalty(self, value):
		self.classifier.penalty = value



