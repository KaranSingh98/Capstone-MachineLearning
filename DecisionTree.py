# Name: Karanveer Singh
# Assignment: Decision Tree Implementation
# Class: CSCI 499 - Capstone
# Instructor: Soumik Dey

import math
import csv
import random
import copy

# attribute values
attrs = {
    1: [1,2,3],
    2: [1,2,3],
    3: [1,2],
    4: [1,2,3],
    5: [1,2,3,4],
    6: [1,2]
}

class Node(object):
    def __init__(self, name, value, entropy, c = None):
        self.name = name
        self.value = value
        self.entropy = entropy
        self.children = []
        self.tested = []
        self.c = c
        self.parent = None

    def add_child(self, obj):
        self.children.append(obj)

    def add_tested(self, attr):
        self.tested.append(attr)

    def add_parent(self, obj):
        self.parent = obj

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.name)+ " = " + repr(self.value)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

    def __eq__(self, other):
        if not isinstance(other, Node):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (self.name == other.name and self.value == other.value and
               self.entropy == other.entropy and self.children == other.children and
               self.tested == other.tested and self.c == other.c)

def entropy(data, attr, attrVals):
    # print(attrVals)
    entropy = 0.0
    total = 0
    classCount = []

    # initialize the classCount array
    for i in range(0, len(attrVals)):
        classCount.append([attrVals[i], 0, 0, 0.0]) # val, class0, class1, entropy
    # print(classCount)

    # count the class occurence for each attribute value
    for row in range(0, len(data)):
        for i in range(0, len(attrVals)):
            # print(classCount[i][3])
            if data[row][attr] == classCount[i][0]:
                if data[row][-1] == 0:
                    classCount[i][1] = classCount[i][1] + 1
                else:
                    classCount[i][2] = classCount[i][2] + 1
        total = total + 1

    # print(classCount)

    # calculate the entropies for the individual attribute values
    for i in range(0, len(classCount)):
        temp = 0.0
        # if classCount[i][1] == 0 and classCount[i][2] != 0:                      # there are no class 0s and there are class 1s
        #     # print('entered')
        #     prob0 = 0.0
        #     prob1 = classCount[i][2] / (classCount[i][1] + classCount[i][2])
        #     temp = ((prob1*math.log2(prob1)))
        # elif classCount[i][1] != 0 and classCount[i][2] == 0:                    # there are class 0s and no class 1s
        #     prob0 = classCount[i][1] / (classCount[i][1] + classCount[i][2])
        #     prob1 = 0.0
        #     temp = ((prob0*math.log2(prob0)))
        # elif classCount[i][1] == 0 and classCount[i][2] == 0:
        #     temp = 0.0
        if classCount[i][1] == 0 or classCount[i][2] == 0:
            temp = 0.0
        else:
            prob0 = classCount[i][1] / (classCount[i][1] + classCount[i][2])
            # print(prob0)
            prob1 = classCount[i][2] / (classCount[i][1] + classCount[i][2])
            # print(prob1)
            prob0 = -prob0*(math.log2(prob0))
            prob1 = -prob1*(math.log2(prob1))
            temp = prob0 + prob1

        # print(temp)

        classCount[i][3] = temp
    # print(classCount)

    # caluculate the final attribute entropy
    for i in range(0, len(classCount)):
        if total > 0:
            if classCount[i][3] != 0.0:
                # print(entropy)
                entropy = entropy + ((((classCount[i][1]+classCount[i][2])/total))*classCount[i][3])

    # print(entropy)
    return entropy

def entropyPerAttr(data, attr, attrVal, val):

    entropy = 0.0
    total = 0
    classCount = [val, 0, 0, 0.0]

    # initialize the classCount array
    # for i in range(0, len(attrVals)):
    #     classCount.append([attrVals[i], 0, 0, 0.0]) # val, class0, class1, entropy
    # # print(classCount)

    # count the class occurence for each attribute value
    for row in range(0, len(data)):

        if data[row][attr] == classCount[0]:
            if data[row][-1] == 0:
                classCount[1] = classCount[1] + 1
            else:
                classCount[2] = classCount[2] + 1
        total = total + 1

    # print(classCount)

    # calculate the entropies for the individual attribute values
    temp = 0.0
    # if classCount[1] == 0 and classCount[2] != 0:                      # there are no class 0s and there are class 1s
    #     # print('entered')
    #     prob0 = 0.0
    #     prob1 = classCount[2] / (classCount[1] + classCount[2])
    #     temp = -1*((prob1*math.log2(prob1)))
    # elif classCount[1] != 0 and classCount[2] == 0:                    # there are class 0s and no class 1s
    #     prob0 = classCount[1] / (classCount[1] + classCount[2])
    #     prob1 = 0.0
    #     temp = -1*((prob0*math.log2(prob0)))
    # elif classCount[1] == 0 and classCount[2] == 0:
    #     temp = 0.0
    if classCount[1] == 0 or classCount[2] == 0:
        temp = 0.0
    else:
        prob0 = classCount[1] / (classCount[1] + classCount[2])
        prob1 = classCount[2] / (classCount[1] + classCount[2])
        temp = -1*(((prob0*math.log2(prob0)) + (prob1*math.log2(prob1))))

    classCount[3] = temp

    # print(classCount)

    # # caluculate the final weighted attribute entropy
    # for i in range(0, len(classCount)):
    #     if total > 0:
    #         entropy = entropy + (((classCount[i][1]+classCount[i][2])/total))*classCount[i][3]

    return classCount[3]

def loadMonks(file):
  with open(file) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=" ")
    data=[]
    for row in csvReader:
      row.pop()
      row.pop(0)
      # row[0], row[6] = row[6], row[0]
      row.append(row.pop(0))

      d=[]

      for r in row:
        d.append(int(r))
        # print(d)
      data.append(d)
      # print(data)
    return data

def stratify(data, size):

    c0, c1, total = 0, 0, 0

    for i in range(0, len(data)):
        if data[i][-1] == 1:
            c1 = c1 + 1
        else:
            c0 = c0 + 1
        total = total + 1

    c0size = int((c0/total) * size)
    c1size = int((c1/total) * size)

    # print("C0 size is " + str(c0size))
    # print("C1 size is " + str(c1size))

    newData = []
    curr0, curr1 = 0, 0
    for j in range(0, len(data)):
        if (data[j][-1] == 1) and (curr1 < c1size):
            # temp = curr1
            curr1 = curr1 + 1
            newData.append(data[j])
            # print(j)
            # print(curr1)
        elif data[j][-1] == 0 and curr0 < c0size:
            curr0 = curr0 + 1
            newData.append(data[j])
    return newData

def createRoot(data):
    min = [1.1, 0]    #1.1 is initially set because entropy can't be greater than 1
                      # so, it accounts for when the min entropy would be 1.0
    for i in range(0, len(data[0])-1):
        e = entropy(data, i, attrs[i+1])
        if e < min[0]:
            min[0] = e
            min[1] = i+1

    # root = Node("Attr"+str(min[1]) + " = " + str(min[0]) + " (" + str(min[0]) + ")", min[1], min[0])
    root = Node(min[1], min[1], min[0])
    # print(root.value)

    for i in range(0, len(attrs[min[1]])):
        newData = []
        for j in range(0, len(data)):
            if data[j][root.value - 1] == i+1:
                newData.append(data[j])

        e = entropyPerAttr(newData, root.value-1, attrs[root.value], i+1)
        # print(str(i+1) + " = " + str(e))
        if e == 0.0:
            leaf = Node("("+str(i+1)+") class = " + str(newData[0][-1]), i+1, e, newData[0][-1])
            leaf.add_parent(root)
            root.add_child(leaf)
        else:
            # print(e)
            # child = Node("Attr" + str(min[1]) + " = " + str(i+1) + " (" + str(e) + ")" , i+1, e)
            child = Node(min[1], i+1, e)
            child.add_tested(min[1])
            child.add_parent(root)
            root.add_child(child)

    return root

def buildTree(root, data):
    # print(root)
    if root.c is not None:
        # print(root)
        # leaf = Node("("+str(root.value)+") class = " + str(data[0][-1]), root.value, root.entropy, data[0][-1])
        # root = leaf
        return root

    if len(data) == 0:
        # print(root)
        c = random.randint(0,1)
        leaf = Node("(" + str(root.value) + ") class = " + str(c), root.value, root.entropy, c)
        leaf.add_parent(root)
        root.add_child(leaf)
        return root

    if len(root.children) > 0:
        for i in range(0, len(root.children)):
            newData = []
            for j in range(0, len(data)):
                if data[j][int(root.name)-1] == root.children[i].value:
                    newData.append(data[j])
            # if len(newData) > 0:
            buildTree(root.children[i], newData)
            # else:
                # root.children.remove(root.children[i])
    else:
        # calculate info gain
        newData = []
        for w in range(0, len(data)):
            if data[w][int(root.name)-1] == root.value:
                newData.append(data[w])

        # print(newData)
        # print(root)
        min = [1.0, 0, 0.0]                 # entropy, attr, information gain
        infoGain = 0.0
        for k in range(0, len(data[0])-1):
            # print(root.tested)
            # if k+1 not in root.tested:      # if the attribute has already been tested on this branch
                # print(root.value)
            e = entropy(newData, k, attrs[k+1])
            infoGain = root.entropy - e
            # print(str(k+1) + " " + str(infoGain))
            # print(root)
            if infoGain >= min[2]:
                # print(root)
                # print(k+1)
                # print((infoGain))
                min[0] = e
                min[1] = k+1
                min[2] = infoGain
    # print(min[1])

        # print("info gain for Attr " + str(min[1]) + " = " + str(min[2]))

        # if min[1] == 0:
        #     # print(newData)
        #     # print(root)
        #     leaf = Node("("+str(root.value)+") class = "  + str(data[0][-1]), root.value, min[0], data[0][-1])
        #     root.add_child(leaf)
        #     # root = leaf

        if min[2] == 0.0:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            leaf = Node("("+str(root.value)+") class = "  + str(data[0][-1]), root.value, min[0], data[0][-1])
            leaf.add_parent(root)
            root.add_child(leaf)
        else:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            # node = Node("Attr" + str(min[1]) +" = " + str(min[0]) +" (" + str(min[0]) + ")" , min[1], min[0])
            node = Node(min[1], min[1], min[0])
            node.tested = root.tested
            node.add_tested(min[1])
            node.add_parent(root);
            childData = []
            # print(node)
            for i in range(0, len(attrs[min[1]])):
                for j in range(0, len(newData)):
                    if newData[j][min[1]-1] == attrs[min[1]][i]:
                        childData.append(data[j])

                e = entropyPerAttr(childData, min[1]-1, attrs[min[1]], i+1)
                child = Node(min[1], i+1, e)
                child.tested = node.tested
                child.add_parent(node)
                node.add_child(child)
            # print(node)
            root.add_child(node)
            # print(childData)
            # print(root)
            # print(childData)
            # print(newData)
        return buildTree(root.children[0], newData)

def classify(root, row):
    # if root.c is not None or len(root.children) == 0:
    #     return root.c

    if len(root.children) == 0:
        return root.c

    elif len(root.children) == 1:
        return classify(root.children[0], row)

    for i in range(0, len(root.children)):
        if row[root.value-1] == root.children[i].value:
            return classify(root.children[i], row)

def evaluate(tree, test):
    prediction = 0
    correct = 0

    for i in range(0, len(test)):
        prediction = classify(tree, test[i])

        # if prediction == None:
        #     prediction = random.randint(0,1)

        if prediction == test[i][-1]:
            correct = correct + 1

    return (correct/len(test))

def replaceNode(tree, node, leaf):

    if tree.c is not None:
        return

    # print(node)

    for i in range(0, len(tree.children)):
        # print(tree.children[i])
        if tree.children[i] == node and tree.children[i].parent == node.parent:
            # print('here')
            tree.children[i] = leaf
        else:
            replaceNode(tree.children[i], node, leaf)

def countClass(root, count):

    if len(root.children) == 0:
        if root.c == 1:
            count[1] = count[1] + 1
        elif root.c == 0:
            count[0] = count[0] + 1

    for i in range(0, len(root.children)):
        countClass(root.children[i], count)

def postPrune(tree, root, val, parent = None, childNum = 0):

    if root.c is not None:
        return tree

    if len(root.children) > 0:
        allLeaf = True
        for i in range(0, len(root.children)):
            # print(root)
            if root.children[i].c is None:
                allLeaf = False
                break

        if allLeaf == False:
            for j in range(0, len(root.children)):
                # print(root.children[j])
                # print("NODE")
                # print(root)
                postPrune(tree, root.children[j], val, root.children[j].parent)
                # print("TEMP")
                # print(temp)
                # break
        else:
            # print("TREE")
            # print(tree)
            # if childNum == len(parent.children):
                # print('ZERO')
            # print("NODE TO CHECK")
            # print(len(parent.children))
            # print(childNum)

            if parent.parent.parent == tree:  # if the current node is the immediate child of the tree root
                parent = parent.parent

            # print(parent)

            # c1 = 0
            # c0 = 0
            count = [0,0]

            countClass(parent,count)
            # print(count)

            if count[0] > count[1]:
                cl = 0
            elif count[1] > count[0]:
                cl = 1
            else:
                cl = random.randint(0,1)

            leaf = Node("class = " + str(cl), parent.value, None, cl)
            # print("\nLEAF")
            # print(leaf)

            originalTree = copy.deepcopy(tree)
            newTree = copy.deepcopy(tree)
            # print('before')
            # print(temp)
            replaceNode(newTree, parent, leaf)
            # print(tree)
            # print('after')
            # print(newTree)

            originalAcc = evaluate(originalTree, val)
            newAcc = evaluate(newTree, val)

            # print("original acc = " + str(originalAcc))
            #
            # print("new acc = " + str(newAcc))

            if newAcc >= originalAcc:
                # print('HERE')
                replaceNode(tree, parent, leaf)

def prePruneHelper(tree, root, data, val, sampleLimit):
    # print(len(data))
    if root.c is not None or len(data) == 0:
        return root

    if len(root.children) > 0:
        for i in range(0, len(root.children)):
            newData = []
            for j in range(0, len(data)):
                if data[j][int(root.name)-1] == root.children[i].value:
                    newData.append(data[j])

            prePruneHelper(tree, root.children[i], newData, val, sampleLimit)
    else:
        # calculate info gain
        newData = []
        for w in range(0, len(data)):
            if data[w][int(root.name)-1] == root.value:
                newData.append(data[w])



        min = [1.0, 0, 0.0]                 # entropy, attr, information gain
        infoGain = 0.0
        for k in range(0, len(data[0])-1):

            e = entropy(newData, k, attrs[k+1])
            infoGain = root.entropy - e

            if infoGain >= min[2]:
                min[0] = e
                min[1] = k+1
                min[2] = infoGain

        # print(root)
        # print(len(newData))


        #
        # print(prevTree)

        if len(newData) <= sampleLimit:
            # prevTree = copy.deepcopy(tree)

            c0, c1 = 0, 0
            for z in range(0, len(newData)):
                if newData[z][-1] == 1:
                    c1 = c1 + 1
                else:
                    c0 = c0 + 1

            if c1 > c0:
                cl = 1
            elif c0 > c1:
                cl = 0
            else:
                cl = random.randint(0,1)

            leaf = Node("("+str(root.value)+") class = "  + str(cl), root.value, min[0], cl)

            replaceNode(tree, root, leaf)
            return

        if min[2] == 0.0:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            leaf = Node("("+str(root.value)+") class = "  + str(data[0][-1]), root.value, min[0], data[0][-1])
            leaf.add_parent(root)
            root.add_child(leaf)
        else:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            # node = Node("Attr" + str(min[1]) +" = " + str(min[0]) +" (" + str(min[0]) + ")" , min[1], min[0])
            node = Node(min[1], min[1], min[0])
            node.tested = root.tested
            node.add_tested(min[1])
            node.add_parent(root);
            childData = []
            # print(node)
            for i in range(0, len(attrs[min[1]])):
                for j in range(0, len(newData)):
                    if newData[j][min[1]-1] == attrs[min[1]][i]:
                        childData.append(data[j])

                e = entropyPerAttr(childData, min[1]-1, attrs[min[1]], i+1)
                child = Node(min[1], i+1, e)
                child.tested = node.tested
                child.add_parent(node)
                node.add_child(child)
            # print(node)
            root.add_child(node)
            # print(childData)
            # print(root)
            # print(childData)
            # print(newData)
        # prevAcc = evaluate(prevTree, val)
        # newAcc = evaluate(tree, val)
        # print(tree)
        #
        # print("prev acc is "+ str(prevAcc))
        # print("new acc is "+ str(newAcc))


        # if prevAcc >= newAcc:
        #     replaceNode(tree, root, leaf)
        #     return
        # else:
        return prePruneHelper(tree, root.children[0], newData, val, sampleLimit)

def prePrune(tree, train, val):

    max = 0.0
    size = 0
    best = None
    for i in range(0, len(train)-1):
         temp = copy.deepcopy(tree)
         prePruneHelper(temp, temp, train, val, i)
         acc = evaluate(temp, val)
         if acc >= max:
             max = acc
             size = i
             best = temp
    return best

def buildTreeWithDepth(root, data, depth, curDepth = 0):

    # print(curDepth)

    if root.c is not None:
        return root

    if len(data) == 0:
        cl = random.randint(0,1)
        leaf = Node("("+str(root.value)+") class = "  + str(cl), root.value, None, cl)
        leaf.add_parent(root)
        root.add_child(leaf)
        return root

    if len(root.children) > 0:
        for i in range(0, len(root.children)):
            newData = []
            for j in range(0, len(data)):
                if data[j][int(root.name)-1] == root.children[i].value:
                    newData.append(data[j])
            buildTreeWithDepth(root.children[i], newData, depth, curDepth)
    else:
        # calculate info gain
        newData = []
        for w in range(0, len(data)):
            if data[w][int(root.name)-1] == root.value:
                newData.append(data[w])

        min = [1.0, 0, 0.0]                 # entropy, attr, information gain
        infoGain = 0.0
        for k in range(0, len(data[0])-1):
            e = entropy(newData, k, attrs[k+1])
            infoGain = root.entropy - e

            if infoGain >= min[2]:
                min[0] = e
                min[1] = k+1
                min[2] = infoGain

        if curDepth+1 == depth:
            # print("HERE")
            c0, c1 = 0, 0
            for z in range(0, len(newData)):
                if newData[z][-1] == 1:
                    c1 = c1 + 1
                else:
                    c0 = c0 + 1

            if c1 > c0:
                cl = 1
            elif c0 > c1:
                cl = 0
            else:
                cl = random.randint(0,1)

            leaf = Node("("+str(root.value)+") class = "  + str(cl), root.value, min[0], cl)

            leaf.add_parent(root)
            root.add_child(leaf)
            return

        if min[2] <= 0.0:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            leaf = Node("("+str(root.value)+") class = "  + str(data[0][-1]), root.value, min[0], data[0][-1])
            leaf.add_parent(root)
            root.add_child(leaf)
            return buildTreeWithDepth(root.children[0], newData, depth, curDepth+1)
        else:
            # print("info gain for Attr " + str(min[1]) + " = " + str(root.value) +" : "+ str(min[2]))
            # node = Node("Attr" + str(min[1]) +" = " + str(min[0]) +" (" + str(min[0]) + ")" , min[1], min[0])
            node = Node(min[1], min[1], min[0])
            node.tested = root.tested
            node.add_tested(min[1])
            node.add_parent(root);
            childData = []
            # print(node)
            for i in range(0, len(attrs[min[1]])):
                for j in range(0, len(newData)):
                    if newData[j][min[1]-1] == attrs[min[1]][i]:
                        childData.append(data[j])

                e = entropyPerAttr(childData, min[1]-1, attrs[min[1]], i+1)
                child = Node(min[1], i+1, e)
                child.tested = node.tested
                child.add_parent(node)
                node.add_child(child)
            # print(node)
            root.add_child(node)

            return buildTreeWithDepth(root.children[0], newData, depth, curDepth + 1)

# LOAD DATA

unpruneTraining1 = loadMonks("monks-1.train")
data1 = loadMonks("monks-1.train")
trainSize = int(len(data1)*.8)
train1 = stratify(data1, trainSize)
val1 = stratify(data1, len(data1) - trainSize + 1)
test1 = loadMonks("monks-1.test")

unpruneTraining2 = loadMonks("monks-2.train")
data2 = loadMonks("monks-2.train")
trainSize = int(len(data2)*.8)
train2 = stratify(data2, trainSize)
val2 = stratify(data2, len(data2) - trainSize + 1)
test2 = loadMonks("monks-2.test")

unpruneTraining3 = loadMonks("monks-3.train")
data3 = loadMonks("monks-3.train")
trainSize = int(len(data3)*.8)
train3 = stratify(data3, trainSize)
val3 = stratify(data3, len(data3) - trainSize + 1)
test3 = loadMonks("monks-3.test")

 # BUILD THE TREES

tree1 = createRoot(unpruneTraining1)
buildTree(tree1, unpruneTraining1)

tree2 = createRoot(unpruneTraining2)
buildTree(tree2, unpruneTraining2)

tree3 = createRoot(unpruneTraining3)
buildTree(tree3, unpruneTraining3)

# get accuracies

acc1 = evaluate(tree1, test1)
acc2 = evaluate(tree2, test2)
acc3 = evaluate(tree3, test3)

print("UNPRUNED DECISION TREE RESULTS")
print("\nMonks 1 = " + str(acc1))
print("Monks 2 = " + str(acc2))
print("Monks 3 = " + str(acc3) + "\n")

# POST PRUNING DECISION TREES

postPrune1 = copy.deepcopy(tree1)
# buildTree(postPrune1, train1)
postPrune(postPrune1, postPrune1, val1)

postPrune2 = copy.deepcopy(tree2)
# buildTree(postPrune2, train2)
postPrune(postPrune2, postPrune2, val2)

postPrune3 = copy.deepcopy(tree3)
# buildTree(postPrune3, train3)
postPrune(postPrune3, postPrune3, val3)

print("POST-PRUNING DECISION TREE RESULTS\n")
print("\nMonks 1 = " + str(evaluate(postPrune1, test1)))
print("Monks 2 = " + str(evaluate(postPrune2, test2)))
print("Monks 3 = " + str(evaluate(postPrune3, test3))+"\n")

# PRE PRUNING DECISION TREES

prePrune1 = createRoot(train1)
prePrune1 = prePrune(prePrune1, train1, val1)

prePrune2 = createRoot(train2)
prePrune2 = prePrune(prePrune2, train2, val2)

prePrune3 = createRoot(train3)
prePrune3 = prePrune(prePrune3, train3, val3)

print("\nPRE-PRUNING DECISION TREE RESULTS\n")
print("\nMonks 1 = " + str(evaluate(prePrune1, test1)))
print("Monks 2 = " + str(evaluate(prePrune2, test2)))
print("Monks 3 = " + str(evaluate(prePrune3, test3)) + "\n")

# HEIGHT RESTRICTION ON DECISION TREES

print("\nHEIGHT RESTRICTED TREE RESULTS\n")
depth = 8

height1 = createRoot(unpruneTraining1)
buildTreeWithDepth(height1, unpruneTraining1, depth)
print("Monks 1 (height "+ str(depth) + ") = " + str(evaluate(height1, test1)))

height2 = createRoot(unpruneTraining2)
buildTreeWithDepth(height2, unpruneTraining2, depth)
print("Monks 2 (height "+ str(depth) + ") = " + str(evaluate(height2, test2)))

height3 = createRoot(unpruneTraining3)
buildTreeWithDepth(height3, unpruneTraining3, depth)
print("Monks 3 (height "+ str(depth) + ") = " + str(evaluate(height3, test3)))
