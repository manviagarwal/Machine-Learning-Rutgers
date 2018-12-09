from __future__ import division
import inspect
import sys
from collections import Counter
from operator import add
import math
prior = []
prior_2 = []
prior_log = []
prior_log_2 = []
likelihood = []  # contains all digits and the probabilities for each feature of a digit
likelihood_A = []
likelihood_B = []
likelihood_C = []

'''
Raise a "not defined" exception as a reminder 
'''


def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
forground (part of the digit) 
'''


def extract_basic_features(digit_data, width, height):
    features = []
    for row in range(0, height):
        for col in range(0, width):
            # features.append(digit_data[row][col])
            if digit_data[row][col] == 0:
                features.append(False)
            else:
                features.append(True)
    # print(features)
    # print(len(features))
    return features


'''
Extract advanced features that you will come up with 
'''

# if '+' set True else set it to False


def extract_advanced_features(digit_data, width, height):
    features = []
    for row in range(0, height):
        for col in range(0, width):
            if digit_data[row][col] == 0:
                features.append('A')
            elif digit_data[row][col] == 1:
                features.append('B')
            elif digit_data[row][col] == 2:
                features.append('C')
    # print(features)
    # print(len(features))

    return features


def extract_advanced_features_2(digit_data, width, height):
    features = []
    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    for row in range(0, height):
        for col in range(0, width):
            if digit_data[row][col] == 2:
                features.append(True)
            else:
                features.append(False)
    # print(features)
    # print(len(features))

    return features


# if '#' set True else set it to False


def extract_advanced_features_3(digit_data, width, height):
    features = []
    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    for row in range(0, height):
        for col in range(0, width):
            if digit_data[row][col] == 1:
                features.append(True)
            else:
                features.append(False)
    # print(features)
    # print(len(features))

    return features


'''
Extract the final features that you would like to use
'''


# uses compute_statistics_2. compute_class_2, classify_2
def extract_final_features(digit_data, width, height):
    features = []
    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    for row in range(0, height):
        for col in range(0, width):
            if digit_data[row][col] == 1:
                features.append(True)
            else:
                features.append(False)
    # print(features)
    # print(len(features))

    return features


'''
Compute the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training. 
'''


def compute_statistics(data, label, width, height, feature_extractor, percentage=100.0):
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # Percentages:
    samples_number = (percentage / 100) * len(data)
    label_split = label[0:int(samples_number)]
    data_split = data[0:int(samples_number)]

    # prior

    global prior
    global prior_log
    global likelihood

    for i in range(10):
        prior_each_digit = label.count(i)
        prior.append(prior_each_digit/len(label_split))

    # likelihood

    for digit in range(10):

        features = []  # for storing the features

        # get the indices from the labels where each digit is present

        indices = [i for i, x in enumerate(label_split) if x == digit]

        # get the training images corresponding to these indices

        images = list(map(lambda u: data_split[u], indices))

        for image in images:  # images contains all the data for label i

            features.append(feature_extractor(image, width, height))

        likelihood_probs = []  # gives probabilities for each image

        # Number of times each feature took the value true in the training example of label

        for ft in range(width*height):

            # count how many trues are there

            count_true = 0
            count_false = 0

            # count how many trues are there for that feature
            for k in features:
                if k[ft] is True:
                    count_true = count_true + 1

            # count how many false are there for that feature
            for kf in features:
                if kf[ft] is False:
                    count_false = count_false + 1

            # prob = count/len(images)

            # Laplace smoothing - one up smoothing
            ky = 0.0001

            prob = (count_true + ky)/(len(images) + ky*2)

            likelihood_probs.append(prob)  # this gives the probabilities for each feature

        likelihood.append(likelihood_probs)  # append the likelihood probabilities for each digit

    prior_log = list(map(lambda y: math.log(y), prior))


'''
For the given features for a single digit image, compute the class 
'''


def compute_class(features):

    global prior_log
    global likelihood

    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # a list to store the likelihood for each feature -- add prior later -- take log and add later
    # find out the probabilities for each feature|label for each feature in testing data

    likelihood_testing = []  # [[],[],[],[],[],[],[].......]

    for digit in range(10):
        temp = []
        for i in range(len(features)):  # 784 is the length
            if features[i] is True:
                temp.append(likelihood[digit][i])
            if features[i] is False:
                temp.append(1-likelihood[digit][i])
        likelihood_testing.append(temp)

    # compute the log of each feature|label

    temp = []

    for elem in likelihood_testing:
        temp.append(list(map(lambda z: math.log(z), elem)))

    # compute the sum of log of each feature|label

    temp2 = []

    for elem in temp:
        temp2.append(sum(elem))

    likelihood_validation = temp2  # likelihood_validation contains the sum of log of probabilities

    # log posterior probability
    posterior_validation = list(map(add, prior_log, likelihood_validation))

    # take the max of all probabilities
    max_probability = max(posterior_validation)

    predicted = posterior_validation.index(max_probability)

    return predicted


'''
Compute joint probability for all the classes and make predictions for a list
of data
'''


def classify(data, width, height, feature_extractor):

    predicted = []

    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # get the features for each image
    features_validation = []

    for image in data:
        features_validation.append(feature_extractor(image, width, height))

    # call compute class for each and every feature

    for i in features_validation:
        predicted.append(compute_class(i))

    return predicted


def compute_statistics_2(data, label, width, height, feature_extractor, percentage=100.0):
    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # Percentages:
    samples_number = (percentage / 100) * len(data)
    label_split = label[0:int(samples_number)]
    data_split = data[0:int(samples_number)]

    # prior

    global prior_2
    global prior_log_2
    global likelihood_A
    global likelihood_B
    global likelihood_C

    for i in range(10):
        prior_each_digit = label.count(i)
        prior_2.append(prior_each_digit/len(label_split))

    # likelihood

    for digit in range(10):

        features = []  # for storing the features

        # get the indices from the labels where each digit is present

        indices = [i for i, x in enumerate(label_split) if x == digit]

        # get the training images corresponding to these indices

        images = list(map(lambda u: data_split[u], indices))

        for image in images:  # images contains all the data for label i

            features.append(feature_extractor(image, width, height))

        likelihood_probs_A = []  # gives probabilities for each image
        likelihood_probs_B = []  # gives probabilities for each image
        likelihood_probs_C = []  # gives probabilities for each image

        # Number of times each feature took the value A, B or C in the training example of label

        for ft in range(width*height):

            # count how many A's , B's and C's  are there

            count_A = 0
            count_B = 0
            count_C = 0

            # count how many A types are there for that feature
            for k in features:
                if k[ft] == 'A':
                    count_A = count_A + 1

            # count how many B types are there for that feature
            for kf in features:
                if kf[ft] == 'B':
                    count_B = count_B + 1

            # count how many C types are there for that feature
            for k in features:
                if k[ft] == 'C':
                    count_C = count_C + 1

    ########################################################################################

            # Laplace smoothing - one up smoothing
            ky = 0.0001

            prob_A = (count_A + ky)/(len(images) + ky*2)
            prob_B = (count_B + ky)/(len(images) + ky*2)
            prob_C = (count_C + ky)/(len(images) + ky*2)

            likelihood_probs_A.append(prob_A)  # this gives the probabilities for each feature
            likelihood_probs_B.append(prob_B)  # this gives the probabilities for each feature
            likelihood_probs_C.append(prob_C)  # this gives the probabilities for each feature

        likelihood_A.append(likelihood_probs_A)  # append the likelihood probabilities for each digit
        likelihood_B.append(likelihood_probs_B)  # append the likelihood probabilities for each digit
        likelihood_C.append(likelihood_probs_C)  # append the likelihood probabilities for each digit

    prior_log_2 = list(map(lambda y: math.log(y), prior_2))


def compute_class_2(features):

    global prior_log_2
    global likelihood_A
    global likelihood_B
    global likelihood_C


    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # a list to store the likelihood for each feature -- add prior later -- take log and add later
    # find out the probabilities for each feature|label for each feature in testing data

    likelihood_testing = []  # [[],[],[],[],[],[],[].......]

    for digit in range(10):
        temp = []
        for i in range(len(features)):  # 784 is the length
            if features[i] == 'A':
                temp.append(likelihood_A[digit][i])
            if features[i] == 'B':
                temp.append(likelihood_B[digit][i])
            if features[i] == 'C':
                temp.append(likelihood_C[digit][i])

        likelihood_testing.append(temp)

    # compute the log of each feature|label
    temp = []

    for elem in likelihood_testing:
        temp.append(list(map(lambda z: math.log(z), elem)))

    # compute the sum of log of each feature|label

    temp2 = []

    for elem in temp:
        temp2.append(sum(elem))

    likelihood_validation = temp2  # likelihood_validation contains the sum of log of probabilities


    # log posterior probability

    posterior_validation = list(map(add, prior_log_2, likelihood_validation))

    # take the max of all probabilities
    max_probability = max(posterior_validation)

    predicted = posterior_validation.index(max_probability)

    return predicted


'''
Compute joint probability for all the classes and make predictions for a list
of data
'''


def classify_2(data, width, height, feature_extractor):

    predicted = []

    # Your code starts here
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    # get the features for each image
    features_validation = []

    for image in data:
        features_validation.append(feature_extractor(image, width, height))

    # call compute class for each and every feature

    for i in features_validation:
        predicted.append(compute_class_2(i))

    return predicted
