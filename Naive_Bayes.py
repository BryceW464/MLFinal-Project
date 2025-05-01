import math

def tokenize(path):
    #Var holding "spam" or "ham" values to note whether a message was spam or ham
    spamham = []
    #same as spamham var but only for training set
    trainingSpamOrHam = []
    #same as spamham var but only for testing set
    testingSpamOrHam = []
    #list of all messages in the doc
    messages = []
    #dictionary of all words in spam messages : the number of times they appear
    trainingSpam = {}
    #same as trainingSpam but for ham training messages
    trainingHam = {}
    #same as trainingSpam but for spam testing messages
    testingSpam = {}
    #same as trainingSpam but for ham testing messages
    testingHam = {}
    #list of all messages used in the training set
    trainingMessages = []
    #list of all messages used in the test set
    testingMessages = []

    with open(path) as file:
        # eads through the data file, appends each message and whether they are spam/ham to their apropriate lists
        #also gets a count of messages in the file
        messageCount = 0

        for line in file:
            messageCount += 1
            tokens = line.split("\t")
            spamham.append(tokens[0])
            messages.append(sanitize(tokens[1]))

        #Declares that the first 80% of data will be used in training
        trainingCount = round(messageCount*0.8)

        #loops through all the messages and updates message, spamham, and word count variables
        for x in range(messageCount):
            if x < trainingCount:
                trainingSpamOrHam.append(spamham[x])
                trainingMessages.append(messages[x])
            else:
                testingSpamOrHam.append(spamham[x])
                testingMessages.append(messages[x])
            if spamham[x] == "ham":
                for word in messages[x]:
                    if x < trainingCount:
                        if word in trainingHam.keys():
                            trainingHam.update({word : trainingHam[word]+1})
                        else:
                            trainingHam.update({word : 1})
                    else:
                        if word in testingHam.keys():
                            testingHam.update({word : testingHam[word]+1})
                        else:
                            testingHam.update({word : 1})
            elif spamham[x] == "spam":
                for word in messages[x]:
                    if x < trainingCount:
                        if word in trainingSpam.keys():
                            trainingSpam.update({word : trainingSpam[word]+1})
                        else:
                            trainingSpam.update({word : 1})
                    else:
                        if word in testingSpam.keys():
                            testingSpam.update({word : testingSpam[word]+1})
                        else:
                            testingSpam.update({word : 1})
    return trainingSpam, testingSpam, trainingHam, testingHam, trainingSpamOrHam, testingSpamOrHam, trainingMessages, testingMessages

def sanitize(mail):
    #Message sanitization function that just strips each word of some special characters and removes some common english words from consideration
    words = []
    for word in (mail.split(" ")):
        if word.strip('\n?!.()",') not in ["in", "and", "but", "of", "for", "I", "on", "a", "at", "to"]:
            words.append(word.strip('\n?!.()",'))
    return words

def NaiveBayes(messages, SpamData, HamData, DataSpamHam, trainingDataSpamHam):
    #Setting up initial values for stats at the end
    TruePos = 0
    TrueNeg = 0
    FalsePos = 0
    FalseNeg = 0

    #Used to follow the DataSpamHam index corresponding to the message index
    index = 0

    
    ham = 0.0
    spam = 0.0

    #Calculcating the initial probability value of it being spam or ham based on the training data
    for result in trainingDataSpamHam:
        if result == "ham":
            ham += 1.0
        else:
            spam += 1.0
    ham = (ham/len(trainingDataSpamHam))
    spam = (spam/len(trainingDataSpamHam))
    

    #Need to get the total values of all the SpamData and HamData
    totalHamValues = sum(HamData.values())
    totalSpamValues = sum(SpamData.values())
    

    for line in messages:
        #These are just inital values for the Ham and Spam probs based on the training data
        SpamProb = spam
        HamProb = ham
        #Checking for both Spam and Ham probabilities for the given word
        for Word in line:
            #Checks to see if the word exists in the SpamData set
            if Word in SpamData:
                SpamProb = SpamProb * ((1 + SpamData[Word])/totalSpamValues)
            else:
                SpamProb = SpamProb * (1/totalSpamValues)
            #Checks to see if the word exists in the HamData set
            if Word in HamData:
                HamProb = HamProb * ((1 + HamData[Word])/totalHamValues)
            else:
                HamProb = HamProb * (1/totalHamValues)

        #Compare the two values, if ham is higher than spam, it's ham, otherwise it's spam
        #After seeing if it's ham or spam, seeing if it's True Positive, True Negative, False Positive False Negative     
        if HamProb < SpamProb:
            if DataSpamHam[index] == 'spam':
                TruePos += 1
            else:
                FalsePos += 1
        else:
            if DataSpamHam[index] == 'ham':
                TrueNeg += 1
            else:
                FalseNeg += 1
        index += 1

    print(f"Stats for the Naive Baye's results:")
    #       True positive, TP = Correct classification of spam SMS
    #       False positive, FP = Incorrect classification of non-spam SMS as spam,
    #       True negative, TN = Correct classification of non-spam SMS
    #       False negative, FN = Incorrect classification of spam SMS as non-spam
    print(f"True Pos: {TruePos}\nTrue Neg: {TrueNeg}\nFalse Pos: {FalsePos}\nFalse Neg: {FalseNeg}")
     #   Accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(f"Accuracy: {((TruePos + TrueNeg)/(TruePos + TrueNeg + FalseNeg + FalsePos))}")
    #   Precision = TP / (TP + FP)
    print(f"Precision: {(TruePos/(TruePos + FalsePos))}")
    #   Recall = TP / (TP + FN)
    print(f"Recall: {(TruePos/(TruePos + FalseNeg))}")
    #   F1-score = (2 * Precision * Recall) / (Precision + Recall)
    print(f"F1-score: {((2 * (TruePos/(TruePos + FalsePos)) * (TruePos/(TruePos + FalseNeg)) ) / ((TruePos/(TruePos + FalsePos)) + (TruePos/(TruePos + FalseNeg))))}")