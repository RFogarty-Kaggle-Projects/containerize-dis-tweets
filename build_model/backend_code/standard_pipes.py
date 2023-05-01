
""" Convenience functions for generating standard combinations of pipelines (e.g. a standard pipeline for doing text-preprocessing) """

import sklearn as sk
import sklearn.pipeline

import text_clean_pipes as textCleanPipeHelp
import train_pipes as trainPipeHelp

def loadTextPreprocPipeA(removeDuplicateTweets=True):
	_pipeComps = [ ("Convert text to lowercase", textCleanPipeHelp.ConvertToLowerCase() ),
	               ("Expand contractions (e.g. \"we're\" to \"we are\"", textCleanPipeHelp.ReplaceContractions() ),
	               ("Replace repeated punctuation (e.g. '???' to '?')", textCleanPipeHelp.ReplaceRepeatedPunctuation() ),
	               ("Map single-digit numbers to words ('10' to 'ten')", textCleanPipeHelp.MapSingleDigitNumbersToWords() ),
	               ("Remove hyperlinks", textCleanPipeHelp.ReplaceHyperlinks(replacement=" ") ),
	               ("Replace some HTML characters", textCleanPipeHelp.ReplaceSpecialHTMLStrings() ),
	               ("Remove trailing punctuation", textCleanPipeHelp.RemoveTrailingPunctuation() ),
	               ("Remove leading punctuation", textCleanPipeHelp.RemoveLeadingPunctuation() )
	] 

	if removeDuplicateTweets:
		_pipeComps.append(  ("Remove duplicate tweets", textCleanPipeHelp.DropTextDuplicates()) )


	outPipe = sk.pipeline.Pipeline(_pipeComps)
	return outPipe


