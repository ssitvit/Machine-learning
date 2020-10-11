import csv
import statistics as st
import math
def loadCsv(filename):
    lines = csv.reader(open(filename, "r"));
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#The gaussian distribution
def calprob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev*stdev)) * exponent

dataset=loadCsv('pima-indians-diabetes.csv')
size=0.5 #split ratio- the size of testing set
train=[] #training set initialized as an empty list
test=[]  #testing set initialized as an empty list

#splitting the dataset into training and testing
for i in range(int(len(dataset)*size)):
	test.append(dataset[i])
for i in range(int(len(dataset)*size+1),len(dataset)):
	train.append(dataset[i])
print('The lenth of the training set',len(train))
print('The length of the testing set',len(test))

classes=[]
for i in dataset:
	if(i[-1] not in classes):
		classes.append(i[-1]) #list of all unique class values stored in the list named classes
classdict={} #dictionary that is intended to contain all the rows associated with each class value
classdict1={} #dictionary that is intended to contain (mean,standard deviation) of every attribute associated with each class value
classprob={} #dictionary that is intended to contain probabilites of the given sample falling into the class values
#initialization
for i in classes:
	classdict[i]=[]
	classdict1[i]=[]
	classprob[i]=1

#for each class value, all the rows having that class value are appended
for i in classes:
	for row in train:
		if row[-1]==i:
			classdict[i].append(row[:-1])

#for each class value, the tuple(mean, stdev) for each attribute is appended
for classval,datt in classdict.items():
	for col in zip(*datt):
		classdict1[classval].append((st.mean(col),st.stdev(col)))


count=0 #counter to count the number of correctly classified instances
#calculating class probabilites
for row in test:
	for i in classes:
		classprob[i]=1
	for classval,datt in classdict1.items():
		for i in range(len(row[:-1])):
			mean,std=datt[i]
			x=row[i]
			classprob[classval]*=calprob(x,mean,std) #refer gaussian naive bayes theory
	print(classprob," for row ",row)
	#calculating accuracy
	mini=0
	cl=0
	for c,d in classprob.items():
		if d>mini:
			mini=d
			cl=c
	
	if row[-1]==cl:
		count+=1

acc=count/len(test)
print("Accuracy of classifier ",acc)










