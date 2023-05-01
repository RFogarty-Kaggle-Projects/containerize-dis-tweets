

import re
import pandas as pd
import sklearn.base


#Base/template classes
class TextPipeline(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	""" Template class for pipelines dealing with various text-processing

	"""

	#Should rarely require a fit step
	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		raise NotImplementedError("")


#Implementations
class ConvertToLowerCase(TextPipeline):

	def __init__(self, targCol="text"):
		self.targCol = targCol

	def transform(self, inpX):
		outX = inpX.copy()
		outX[self.targCol] = outX[self.targCol].map(lambda x:x.lower())
		return outX

class ReplaceContractions(TextPipeline):
	
	def __init__(self, contractionDict=None, targCol="text"):
		self.contractionDict = _getDecontractMaps() if contractionDict is None else contractionDict
		self.targCol = targCol
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpX):
			outStr = inpX
			for key in self.contractionDict.keys():
				outStr = outStr.replace(key, self.contractionDict[key])
			return outStr
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class ReplaceRepeatedPunctuation(TextPipeline):
	
	def __init__(self,repPunctList=None, targCol="text"):
		_defaultList = ["?",".","!"]
		self.repPunctList = _defaultList if repPunctList is None else repPunctList
		self.targCol = targCol

	def transform(self, inpX):
		outX = inpX.copy()
		for punct in self.repPunctList:
			_regexPart = "[{}]".format(punct) + "{2,}"
			outX[self.targCol] = outX[self.targCol].map( lambda x: re.sub(_regexPart, punct, x)   )
		return outX

class MapSingleDigitNumbersToWords(TextPipeline):
	
	def __init__(self, targCol="text"):
		self.mapDict = self._loadDefDict()
		self.targCol = targCol
	
	def _loadDefDict(self):
		outDict = {"0":"zero", "1":"one", "2":"two", "3":"three", "4":"four",
				   "5":"five", "6":"six", "7":"seven", "8":"eight", "9":"nine"}
		return outDict
	
	def transform(self, inpX):
		outX = inpX.copy()
		pattern = "([ ;:?])([0-9])([ .?;:][^0-9])"
		
		def _mapRegex(inpMatch):
			groups = inpMatch.groups()
			midGroup = self.mapDict[ inpMatch.groups()[1] ]
			return groups[0] + midGroup + groups[-1]
		
		def _mapFunct(inpStr):
			return re.sub(pattern, _mapRegex, inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class ReplaceHyperlinks(TextPipeline):
	
	def __init__(self, targCol="text", replacement=" hyperlink "):
		self.regex = re.compile("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])")
		self.targCol = targCol
		self.replacement = replacement
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpStr):
			return self.regex.sub(self.replacement,inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class ReplaceSpecialHTMLStrings(TextPipeline):
	
	def __init__(self, mapDict=None, targCol="text"):
		self.mapDict = self._getDefaultMapDict() if mapDict is None else mapDict
		self.targCol = targCol
	
	def _getDefaultMapDict(self):
		outDict = {"&amp;":"&", "&gt;":">", "&lt;":"<"}
		return outDict
	
	def transform(self, inpX):
		outX = inpX.copy()
		def _mapFunct(inpStr):
			outStr = inpStr
			for key in self.mapDict:
				outStr = outStr.replace(key, self.mapDict[key])
			return outStr
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class RemoveTrailingPunctuation(TextPipeline):
	
	def __init__(self, punctList=None, targCol="text"):
		self.punctList = self._getDefPunctList() if punctList is None else punctList
		self.targCol = targCol
		
	def _getDefPunctList(self):
		return [".",";",":","?","'", "!"] + [","]
	
	def transform(self, inpX):
		outX = inpX.copy()
		punctPart = "".join(self.punctList)
		pattern = "([\w]+)([" + punctPart + "])"
		def _mapFunct(inpStr):
			return re.sub(pattern, "\\1 \\2",inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX

class RemoveLeadingPunctuation(TextPipeline):
	
	def __init__(self, punctList=None, targCol="text"):
		self.punctList = self._getDefPunctList() if punctList is None else punctList
		self.targCol = targCol
		
	def _getDefPunctList(self):
		return [".",";",":","?","'", "!"]
	
	def transform(self, inpX):
		outX = inpX.copy()
		punctPart = "".join(self.punctList)
		pattern = "([" + punctPart + "])" + "([\w]+)"
		def _mapFunct(inpStr):
			return re.sub(pattern, "\\1 \\2",inpStr)
		
		outX[self.targCol] = outX[self.targCol].map(_mapFunct)
		return outX


class DropTextDuplicates(TextPipeline):

	def __init__(self):
		pass

	def transform(self, inpX):
		outX = inpX.copy()
		return outX.drop_duplicates(["text"])



#Convenience functions

def _getDecontractMaps():
	_decontractMaps = {"i'm":"i am",
	                   "it's":"it is",
	                   "don't":"do not",
	                   "can't":"can not",
	                   "you're":"you are",
	                   "that's":"that is",
	                   "i've":"i have",
	                   "i'll":"i will",
	                   "he's":"he is",
	                   "there's":"there is",
	                   "didn't":"did not",
	                   "i'd":"i did",
	                   "what's":"what is",
	                   "they're":"they are",
	                   "isn't":"is not",
	                   "we're":"we are",
	                   "let's": "let us",
	                   "won't": "will not",
	                   "ain't":"is not",
	                   "we're":"we are",
	                   "reddit's":"reddit is",
	                   "she's":"she is",
	                   "wasn't":"was not",
	                   "haven't":"have not",
	                   "you'll":"you will",
	                   "aren't":"are not",
	                   "we've":"we have",
	                   "wouldn't":"would not",
	                   "you've":"you have",
	                   "here's":"here is",
	                   "it's":"it is",
	                   "shouldn't":"should not",
	                   "who's":"who is",
	                   "we'll":"we will",
	                   "would've":"would have",
	                   "y'all":"you all",
	                   "they've":"they have",
	                   "you'd":"you would",
	                   "doesn't":"does not"
	                  }
	return _decontractMaps


