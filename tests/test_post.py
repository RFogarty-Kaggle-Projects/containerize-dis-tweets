
""" Simple test to check our POST request gets the expected output; assuming the server is running @ local """

import itertools as it
import json
import requests
import unittest

SERVER_URL = "http://127.0.0.1:5000/results"


class TestPostRequest(unittest.TestCase):


	def _runTestFunct(self):
		payload = json.dumps( {"text":self.testTweets} )
		headers = {"Content-Type": "application/json"}
		response = requests.request("POST", SERVER_URL, headers=headers, data=payload)
		return [int(x) for x in response.json().split(",")]

	def testExpectedResponseA(self):
		self.testTweets = [ "There is a fire in London",
		                    "I am hungry",
		                    "A large magnitude earthquake hit XXX",
		                    "I dont like fire" ]

		expVals = [1,0,1,0]
		actVals = self._runTestFunct()

		#We sort of want to print the tweets/classifications out
		for tweet,actVal in it.zip_longest(self.testTweets,actVals):
			clasStr = "Disaster" if actVal==1 else "Not Disaster"
			print("Tweet = {}; classification = {}".format(tweet,clasStr))

		#Pass/fail
		self.assertEqual(expVals, actVals)

if __name__ == '__main__':
	unittest.main()

