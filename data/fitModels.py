import sys
import csv
import numpy as np
import os.path
import json
import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

sys.path.append('../../code/')
import Forest
import Tree

def testModel(roundSplit,XTrain,YTrain,XTest,YTest,model,name):
	print("Fitting", name)
	model.fit(XTrain,YTrain)

	print("Testing ", name)
	start = timeit.default_timer()
	YPredicted = model.predict(XTest)
	end = timeit.default_timer()

	print("Total time: " + str(end - start) + " ms")
	print("Throughput: " + str(len(XTest) / (float(end - start)*1000)) + " #elem/ms")

	print("Saving model")
	if (issubclass(type(model), DecisionTreeClassifier)):
		mymodel = Tree.Tree()
	else:
		mymodel = Forest.Forest()

	mymodel.fromSKLearn(model,roundSplit)

	if not os.path.exists("text"):
		os.makedirs("text")

	with open("text/"+name+".json",'w') as outFile:
		outFile.write(mymodel.str())

	SKPred = model.predict(XTest)
	MYPred = mymodel.predict_batch(XTest)
	accuracy = accuracy_score(YTest, SKPred)
	print("Accuracy:", accuracy)

	# This can now happen because of classical majority vote
	# for (skpred, mypred) in zip(SKPred,MYPred):
	# 	if (skpred != mypred):
	# 		print("Prediction mismatch!!!")
	# 		print(skpred, " vs ", mypred)

	print("Saving model to PKL on disk")
	joblib.dump(model, "text/"+name+".pkl")

	print("*** Summary ***")
	print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
	print(str(len(XTest)) + "\t" + str(len(XTest[0])) + "\t" + str(accuracy) + "\t" + str(mymodel.getAvgDepth()))
	print()

def fitModels(roundSplit,XTrain,YTrain,XTest = None,YTest = None,createTest = False):
	if XTest is None or YTest is None:
		XTrain,XTest,YTrain,YTest = train_test_split(XTrain, YTrain, test_size=0.25)
		createTest = True

	if createTest:
		with open("test.csv", 'w') as outFile:
			for x,y in zip(XTest, YTest):
				line = str(y)
				for xi in x:
					line += "," + str(xi)

				outFile.write(line + "\n")

	# for depth in [1,5,10,15,20,30,50]:
	# 	for num in [1,5,10,15,20,30,50,80,100]:
	for depth in [5,20]:
		for num in [1,10,50,100]:
			testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=num,n_jobs=8,max_depth=depth),"DT_"+str(depth)+"_"+str(num))
