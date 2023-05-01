
import numpy as np
import pandas as pd

import sklearn as sk
import sklearn.base
import sklearn.feature_extraction


class AddBagOfWords(sklearn.base.TransformerMixin):

	def __init__(self, vectKwargs=None, textField="text", colPrefix="bow_"):
		""" Initializer
		
		Args:
			vectKwargs: (dict) Any keywords to pass to "sk.feature_extraction.text.CountVectorizer"
			textField: (str) The field (in input dataframes) we apply the vectorization to
			colPrefix: (str)
				 
		"""
		vectKwargs = dict() if vectKwargs is None else vectKwargs
		self.vectorizer = sk.feature_extraction.text.CountVectorizer(**vectKwargs)
		self.textField = textField
		self.colPrefix = colPrefix

	def fit(self, inpX, y=None):
		self.vectorizer.fit(inpX[self.textField].tolist())
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		countMatrix = self.vectorizer.transform(outX[self.textField].tolist())
		cols = [self.colPrefix + "{}".format(int(x)) for x in range(countMatrix.shape[1])]
		self._bowFrame = pd.DataFrame(countMatrix.todense(), columns=cols)
		self._bowFrame.index = outX.index
		return pd.concat([outX,self._bowFrame],axis=1)

