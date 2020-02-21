## Author: Karanveer Singh
## Professor: Soumik Dey
## Class: CSCI 499 (Capstone)
## Assignemnt: kNN 
## About: Implementation of kNN(weighted and unweighted), as well as with
##        Kd-Tree

import math
import csv
import random

# files where the results will be saved
weightedResults = open("weightedResults.txt", "w")
unWeightedResults = open("unWeightedResults.text", "w")
kdTreeResults = open("kdTreeResults.txt", "w")

monk1UnWeightedResults = open("monks1UnWeightedResults.txt", "w")
monk1WeightedResults = open("monk1s1WeightedResults.txt", "w")
monk1KdTreeResults = open("monks1KdTreeResults.txt", "w")

def normalize(data):
  for i in range(0, len(data)):
    for j in range (0,9):
      data[i][j] = (data[i][j]-1)/9
      
  return data
  
def load(file):
  with open(file) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=';')
    data=[]
    for row in csvReader:
      d=[]
      for r in row[0].split(',')[1:-1]:
        d.append(int(r))
      data.append(d)
      
    return data
    
def euclideanDistance(d1, d2):
  # print(d1, ' ', d2)
  distance = 0
  
  for i in range(0, len(d1)-1):
    distance = distance + ((d1[i]-d2[i])**2)
  
  distance = math.sqrt(distance)
  
  return [distance, d2]
  
def weightedKNN(train, test, k):
  
  correct = 0
  prediction = 2
  
  for s in test:
    distances=[]
    
    class2 = 0
    class4 = 0
    
    for t in train:
      
      predicted = 4
      d = euclideanDistance(s, t)
      distances.append(d)
      
    distances.sort(key=lambda x: x[0])
    
    for i in range(1, k):
      if distances[i][1][-1] == 2:
        class2 = class2 + (1 / (distances[i][0] + 1))
      else:
        class4 = class4 + (1 / (distances[i][0] + 1))
        
      if class2 > class4:
        prediction = 2
      else:
        prediction = 4
    
    if prediction == s[-1]:
      correct = correct + 1
    
  return correct/len(test)
  
def unWeightedKNN(train, test, k):
  correct = 0
  prediction= 2
  
  for s in test:
    distances=[]
    
    class2 = 0
    class4 = 0
    
    for t in train:
      
      d = euclideanDistance(s, t)
      distances.append(d)
      
    distances.sort(key=lambda x: x[0])
    
    for i in range(1, k):
      if distances[i][1][-1] == 2:
        class2 = class2 + 1
      else:
        class4 = class4 + 1
        
      if class2 > class4:
        prediction = 2
      else:
        prediction = 4
    
    if prediction == s[-1]:
      correct = correct + 1
    
  return correct/len(test)
  
# determine optimal k for weight kNN
def determineKforWeighted(train, val, func, file):
  k = 1
  max = 0
  kResults = []
  
  for i in range(1, len(val)):
    cur = func(train, val, i)
    file.write(str(i) + ": " + str(cur) + "\n")
    if cur > max:
      max = cur
      k = i
      kResults=[]
    elif cur == max:
      kResults.append(i)
    
  size = len(kResults)
  
  if size > 0:
    size = int(size/2)
    k = kResults[size]
  
  return k

# determine optimal k for unweighted kNN
def determineKforUnWeightedKNN(train, val, func, file):
  k = 1
  max = 0
  kResults = []
  
  for i in range(1, len(val)):
    cur = func(train, val, i)
    file.write(str(i) + ": " + str(cur) + "\n")
    if cur > max:
      max = cur
      k = i
      kResults=[]
    elif cur == max:
      kResults.append(i)
  
  size = len(kResults)
  
  if size > 0:
    size = int(size/2)
    k = kResults[size]
  
  return k

kdValue = len(train) - 1

def createKDTree(data, depth = 0):
  
  n = int(len(data))
  
  if n <= 0:
    return None
  
  # the axis is the feature around which the tree will split next
  axis = depth % kdValue
  
  # sorting the data around the next feature so we can easily get the median value
  sortedData = sorted(data, key=lambda row:row[axis])
  
  n = int(n/2)
  
  return {
    'value': sortedData[n],
    'leftTree': createKDTree(sortedData[:n], depth + 1),
    'rightTree': createKDTree(sortedData[n + 1:], depth + 1)
  }

def getNearestPoint(root, data, depth = 0, closest=None):
  
  # if leaf of tree is reached
  if root is None:
    return closest
  
  # the current feature being evaluated in the tree  
  axis = depth % kdValue
  
  # next branch to traverse to
  branch = root
  
  # the new closest point 
  cur = closest
  
  # if the current point is closer to the previous closest point
  if closest is None or euclideanDistance(data, closest) > euclideanDistance(data, root['value']):
    cur = root['value']
  # determine where to go next depending on whether the current feature of the row is less than the current point in the tree or not
  if data[axis] < root['value'][axis]:
    branch = root['leftTree']
  else:
    branch = root['rightTree']
    
  return getNearestPoint(branch, data, depth + 1, cur)

def classifyKDTree(tree, test):
  
  correct = 0
  class2 = 0
  class4 = 0
  
  for row in test:
    closest = getNearestPoint(tree, row)
    prediction = closest[-1]
    
    if prediction == 2:
      class2 = class2 + 1
    else:
      class4 = class4 + 1
    
    if prediction == row[-1]:
      correct = correct + 1
      
  return correct/len(test)
  
def classifyKDTreeForMonks(tree, test):
  correct = 0
  class0 = 0
  class1 = 0
  
  for row in test:
    closest = getNearestPoint(tree, row)
    prediction = closest[-1]
    
    if prediction == 0:
      class0 = class0 + 1
    else:
      class1 = class1 + 1
    
    if prediction == row[-1]:
      correct = correct + 1
      
  return correct/len(test)
  
def loadMonks(file):
  with open(file) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=" ")
    data=[]
    for row in csvReader:
      
      row.pop()
      row.pop(0)
      row[0], row[6] = row[6], row[0]
      
      d=[]
      
      for r in row:
        d.append(int(r))
        # print(d)
      data.append(d)
      # print(data)
    return data

def normalizeMonks(data):
  for i in range(0, len(data)):
    for j in range (0,5):
      data[i][j] = (data[i][j]-1)/5
      
  return data

def unWeightedKNNforMonks(train, test, k):
  
  correct = 0
  prediction= 1
  
  for s in test:
    distances=[]
    # print('s is ', s)
    
    class0 = 0
    class1 = 0
    
    for t in train:
      
      d = euclideanDistance(s, t)
      distances.append(d)
      
    distances.sort(key=lambda x: x[0])
    
    for i in range(1, k):
      # print('class is ', distances[i][1][-1])
      if distances[i][1][-1] == 1:
        class1 = class1 + 1
      else:
        class0 = class0 + 1
        
      if class1 > class0:
        prediction = 1
      else:
        prediction = 0
    
    if prediction == s[-1]:
      correct = correct + 1
    
  return correct/len(test)

def weightedKNNforMonks(train, test, k):
  correct = 0
  prediction= 1
  
  for s in test:
    distances=[]
    # print('s is ', s)
    
    class0 = 0
    class1 = 0
    
    for t in train:
      
      d = euclideanDistance(s, t)
      distances.append(d)
      
    distances.sort(key=lambda x: x[0])
    
    for i in range(1, k):
      # print('class is ', distances[i][1][-1])
      if distances[i][1][-1] == 1:
        class1 = class1 + (1/(distances[i][0] + 1))
      else:
        class0 = class0 + (1/(distances[i][0] + 1))
        
      if class1 > class0:
        prediction = 1
      else:
        prediction = 0
    
    if prediction == s[-1]:
      correct = correct + 1
    
  return correct/len(test)
  
# load the datasets
train = load("train.csv")
validate = load("val.csv")
test = load("test.csv")

# normalize the data
train = normalize(train)
validate = normalize(validate)
test = normalize(test)

# Weighted KNN
k = determineKforWeighted(train, validate, weightedKNN, weightedResults)
# print('k used for weighted: ', k)
accuracy = weightedKNN(train, test, k)
# print("Weighted KNN: ", accuracy)
weightedResults.write("Final Result = " + str(k) + ": " + str(accuracy) + "\n")

# UnWeighted KNN
k = determineKforUnWeightedKNN(train, validate, unWeightedKNN, unWeightedResults)
# print('k used for unweighted: ', k)
accuracy = unWeightedKNN(train, test, k)
# print("UnWeighted KNN: ", accuracy)
unWeightedResults.write("Final Result = " + str(k) + ": " + str(accuracy) + "\n")

# Create the Kd-Tree
# n = int(len(train)/2)
# n
# print(train[n])
kdTree = createKDTree(train)
accuracy = classifyKDTree(kdTree, test)
kdTreeResults.write('Kd-Tree Accuracy: ' + str(accuracy))

monks1 = loadMonks("monks-1.train")
monks1 = normalizeMonks(monks1)
# print(monks1)

monksTest = loadMonks("monks-1.test")
monksTest = normalizeMonks(monksTest)
# print(monksTest)

#split monks1 into 80-20 train-validate split
# monks1 = random.shuffle(monks1)
# print(len(monks1))

monkTrainSize = int(len(monks1)*.80)
# print('80 length is : ', monkTrainSize)
monkValSize = int(len(monks1)*.20)
# print('20 length is: ', monkValSize)

monks1Train = monks1[:monkTrainSize]
# print(monks1Train)
monks1Val = monks1[monkValSize:]

# unweighted kNN on monks-1
k = determineKforUnWeightedKNN(monks1Train, monks1Val, unWeightedKNNforMonks, monk1UnWeightedResults)
# print("k used for monks1 unwieighted kNN: ", k)
accuracy = unWeightedKNNforMonks(monks1Train, monksTest, k)
# print("Unweighted kNN on Monks 1: ", accuracy)
monk1UnWeightedResults.write("Final Result = " + str(k) + ": " + str(accuracy) + "\n")

# weighted kNN on monks-1
k = determineKforWeighted(monks1Train, monks1Val, weightedKNNforMonks, monk1WeightedResults)
# print('k used for monks 1 weighted kNN: ', k)
accuracy = weightedKNNforMonks(monks1Train, monksTest, k)
# print("Weighted kNN on Monks 1: ", accuracy)
monk1WeightedResults.write("Final Result = " + str(k) + ": " + str(accuracy) + "\n")

# kNN with Kd-Tree on monks-1
kdValue = len(monks1Train) - 1
kdTree = createKDTree(monks1Train)
accuracy = classifyKDTreeForMonks(kdTree, monksTest)
monk1KdTreeResults.write('kd-Tree Accuracy: ' + str(accuracy))
# print('Kd-Tree: ', accuracy)

# close files used to store results
weightedResults.close()
unWeightedResults.close()
kdTreeResults.close()
monk1WeightedResults.close()
monk1UnWeightedResults.close()
monk1KdTreeResults.close()

