import re
import random
import string
import numpy as np
import pandas as pd
import math
import sys


def get_data_from_wiki():
    """
    Method to read data from the text file and have it in 15 word sentences
    :return:
    """
    with open('wiki_english.txt') as f:
        eng_str = f.read()
    eng_str.replace("|", "")
    eng_str = re.sub(r'^\w', '', eng_str)
    wordlist_english = eng_str.split(" ")

    with open('wiki_dutch.txt') as f:
        dutch_str = f.read()
    dutch_str.replace("|", "")
    dutch_str = re.sub(r'^\w', '', dutch_str)
    wordlist_dutch = dutch_str.split(" ")

    def split(a, n):
        """
        Gets the sentence to be of 15 words
        :param a:
        :param n:
        :return:
        """
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    english_samples = list(split(wordlist_english, len(wordlist_english) // 15))
    dutch_samples = list(split(wordlist_dutch, len(wordlist_dutch) // 15))

    english_sentences = []
    dutch_sentences = []

    for line in english_samples:
        l = " ".join(line)
        english_sentences.append("en| " + l)

    for line in dutch_samples:
        l = " ".join(line)
        dutch_sentences.append("nl| " + l)

    all_sentences = english_sentences + dutch_sentences
    random.shuffle(all_sentences)

    with open('train_input.dat', 'w') as f:
        for line in all_sentences:
            res_str = re.sub(r"\n", " ", line)
            f.writelines(res_str + '\n')


def preprocess_avg_word_length(file):
    """
    preprocessing for avg length of  words in a sentence
    :return:
    """
    lines = file.readlines()
    avg = []
    for line in lines:
        mod_string = line[4:]
        temp = [len(ele) for ele in mod_string.split()]
        avg_length = 0 if len(temp) == 0 else round((float(sum(temp)) / len(temp)), 2)
        if avg_length >= 4 and avg_length <= 6.00:
            avg.append(1)
        else:
            avg.append(0)

    return avg


def preprocess_articles(file):
    """
    Preprocesses the dutch articles in a sentence
    :param file:
    :return:
    """
    lines = file.readlines()
    articles = ['de', 'De', 'het', 'Het', 'een', 'Een']
    article_counter_list = [0 for i in range(len(lines))]
    i = 0
    for sentence in lines:
        article_counter = 0
        for word in sentence.split():
            if word in articles:
                article_counter += 1
        if article_counter > 1:
            article_counter_list[i] = 0
        else:
            article_counter_list[i] = 1
        i += 1
    return article_counter_list


def preprocess_punctuations(file):
    """
    Preprocesses the number of punctuations in a sentence
    :param file:
    :return:
    """
    lines = file.readlines()

    punctuation_counter_list = [0 for _ in range(len(lines))]
    i = 0
    for sentence in lines:
        punctuation_counter = 0
        for punctuation in sentence:
            if punctuation in string.punctuation:
                punctuation_counter += 1
        if punctuation_counter > 3:
            punctuation_counter_list[i] = 1
        else:
            punctuation_counter_list[i] = 0

        i += 1
    return punctuation_counter_list


def preprocess_prepositions(file):
    """
    Preprocesses english prepositions
    :param file:
    :return:
    """
    lines = file.readlines()
    prepositions = ['of', 'Of', 'on', 'On', 'in', 'In', 'for', 'For', 'to', 'To', 'above', 'Above', 'below', 'Below',
                    'over', 'Over']
    preposition_counter_list = [0 for i in range(len(lines))]
    i = 0
    for sentence in lines:
        preposition_counter = 0
        for word in sentence.split():
            if word in prepositions:
                preposition_counter += 1

        if preposition_counter > 2:
            preposition_counter_list[i] = 1
        else:
            preposition_counter_list[i] = 0

        i += 1
    return preposition_counter_list


def preprocess_vowel_pairs(file):
    """
    checks for vowel pairs in a sentence
    :param file:
    :return:
    """
    lines = file.readlines()
    vowels = ['a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'U', 'u']
    vowel_pair_exists_list = []
    i = 0
    for sentence in lines:
        vowel_pairs = 0
        for word in sentence.split():
            for i in range(len(word) - 1):
                if word[i] in vowels and word[i + 1] in vowels and word[i] == word[i + 1]:
                    vowel_pairs += 1
        if vowel_pairs > 3:
            vowel_pair_exists_list.append(1)
        else:
            vowel_pair_exists_list.append(0)
        i += 1
    return vowel_pair_exists_list


def create_target_column(file):
    """
    Gets the target column by checking first 2 letters of sentence
    :param file:
    :return:
    """
    target_column = []
    lines = file.readlines()
    for line in lines:
        if line[:2] == "en":
            target_column.append(1)
        else:
            target_column.append(0)
    return target_column


def create_dat(avg_list, articles, prepositions, punctuations, vowel_pairs, target_column):
    """
    Creates dat file with true and false
    :param avg_list:
    :param articles:
    :param prepositions:
    :param punctuations:
    :param vowel_pairs:
    :param target_column:
    :return:
    """
    with open('dataset.dat', 'w') as f:
        f.writelines(map("{},{},{},{},{},{}\n".format, avg_list, articles,
                         prepositions, punctuations, vowel_pairs, target_column))
        f.close()


def create_testing_dat(avg_list, articles, prepositions, punctuations, vowel_pairs):
    """
    Creating testing dat file without column
    :param avg_list:
    :param articles:
    :param prepositions:
    :param punctuations:
    :param vowel_pairs:
    :return:
    """
    with open('testing_dataset.dat', 'w') as f:
        f.writelines(map("{},{},{},{},{}\n".format, avg_list, articles,
                         prepositions, punctuations, vowel_pairs))
        f.close()


def convert_csv():
    """
    Replacing 1 with True and 0 with False
    :return:
    """
    with open('dataset.dat', 'r') as f:
        w = re.sub(",", " ", f.read())
        y = re.sub('1', "True", w)
        z = re.sub('0', "False", y)
        with open('dataset.dat', 'w') as f1:
            f1.write(z)

    all_data = pd.read_csv('dataset.dat', sep=" ", header=None)
    return all_data


def convert_test_csv():
    """
    Converting the test csv to true and false
    :return:
    """
    with open('testing_dataset.dat', 'r') as f:
        w = re.sub(",", " ", f.read())
        y = re.sub('1', "True", w)
        z = re.sub('0', "False", y)
        with open('testing_dataset.dat', 'w') as f1:
            f1.write(z)

    all_data = pd.read_csv('testing_dataset.dat', sep=" ", header=None)
    return all_data


def entropy_calculator(column):
    """
    Calculating the entropy of a column
    :param column:
    :return:
    """
    counts = np.bincount(column)
    probabilities = counts / len(column)
    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    return -entropy


def information_gain_calculator(data, split_name, target_name):
    """
    Calculating the information gain
    :param data:
    :param split_name:
    :param target_name:
    :return: info gain, left split array, right split array
    """
    original_entropy = entropy_calculator(data[target_name])
    left_split = data[data[split_name] == 0]
    right_split = data[data[split_name] == 1]
    factor = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0])
        factor += prob * entropy_calculator(subset[target_name])
    return original_entropy - factor, left_split, right_split


def decision_tree_helper(all_data, eliminated_col):
    """
    Function to help calculate splits at 2 levels
    :param all_data:
    :param eliminated_col:
    :return:
    """
    information_gains = {}
    ig = float('-inf')
    split_col = -1
    ls, rs = pd.DataFrame(), pd.DataFrame()
    for col in range(0, 5):
        if col in eliminated_col:
            continue
        information_gain, left_split, right_split = information_gain_calculator(all_data, col, 5)
        information_gains[col] = information_gain
        if ig < information_gain:
            ig = information_gain
            split_col = col
            ls = left_split
            rs = right_split
    return split_col, rs, ls


def get_class_name(data):
    """
    Getting the dominating class group
    :param data:
    :return:
    """
    en_count = 0
    du_count = 0
    for line in data[5]:
        if line == True:
            en_count += 1
        else:
            du_count += 1

    if en_count > du_count:
        return True
    else:
        return False


def create_decision_tree(call_depth, data_list, ec, f):
    """
    Recursively creates decision tree
    :param call_depth: level of tree
    :param data_list: current list
    :return:
    """
    flag = find_prob(data_list)
    if len(data_list) >= 10 and call_depth <= 4 and not flag and len(ec) < 5:
        attribute, best_right, best_left = decision_tree_helper(data_list, ec)
        ec.append(attribute)
        f.write('\n' + '\t' * (call_depth + 2) + 'if data_list.iloc[i][' + str(attribute) + '] == 0:')
        l = create_decision_tree(call_depth + 1, best_left, ec, f)
        if not l:
            f.write('\n' + '\t' * (call_depth + 3) + 'classifications.append([' + str(get_class_name(best_left)) + '])')
        f.write('\n' + '\t' * (call_depth + 2) + 'else:')
        r = create_decision_tree(call_depth + 1, best_right, ec, f)
        if not r:
            f.write(
                '\n' + '\t' * (call_depth + 3) + 'classifications.append([' + str(get_class_name(best_right)) + '])')
        return True
    return False


def find_prob(data_list):
    """
    Finds is probability for decision tree
    :param data_list:
    :return:
    """
    p_en_sum = 0
    p_nl_sum = 0

    for i in data_list[5]:
        if i == True:
            p_en_sum += 1
        else:
            p_nl_sum += 1

    p_eng = p_en_sum / len(data_list)
    p_dutch = p_nl_sum / len(data_list)

    if p_dutch > 0.89 or p_eng > 0.89:
        return True
    else:
        return False


def write_classifier(train_data_list, file_name, classifier_filename):
    """
    Writes classifier file
    :param train_data_list:
    :param file_name:
    :param classifier_filename:
    :return:
    """
    with open(classifier_filename, 'w') as f:
        f.write('''import csv
import re
import pandas as pd
import decisiontree\n\n

def create_pd(test_data):
\tdata = pd.read_csv(test_data, sep=" ", header=None)
\treturn data\n\n
def classify(data_list):
\tclassifications = []
\tfor i in range(len(data_list)):''')
        create_decision_tree(0, train_data_list, [], f)
        f.write('''\n\treturn classifications\n\n
def write_classification_csv(classifications):
\twith open('dt_classifications.csv', 'w') as f:
\t\tfo = csv.writer(f)
\t\tfor i in classifications:
\t\t\tif i[0] == True:
\t\t\t\tfo.writerow(['en'])
\t\t\t\tprint("en")
\t\t\telse:
\t\t\t\tfo.writerow(['nl'])
\t\t\t\tprint("nl")
\tf.close()\n\n
def main(traindata):
\tdecisiontree.main({},"",False)
\tdata = create_pd('testing_dataset.dat')
\tclasses = classify(data)
\twrite_classification_csv(classes)\n\n
if __name__ == \'__main__\':
\tmain()'''.format("traindata"))
    f.close()


def dt_helper(all_data, eliminated_col):
    """
    Function to help calculate splits at 2 levels
    :param all_data:
    :param eliminated_col:
    :return:
    """
    information_gains = {}
    ig = float('-inf')
    split_col = -1
    ls, rs = pd.DataFrame(), pd.DataFrame()
    for col in range(0, 5):
        if col in eliminated_col:
            continue
        information_gain, left_split, right_split = information_gain_calculator(all_data, col, 5)
        information_gains[col] = information_gain
        if ig < information_gain:
            ig = information_gain
            split_col = col
            ls = left_split
            rs = right_split

    left_mistakes = 0
    right_mistakes = 0

    left_positions = []
    right_positions = []
    if get_class_name(ls) == True:
        for counter,i in enumerate(ls[5]):
            if i == False:
                left_positions.append(counter)
                left_mistakes += 1
    else:
        for counter,i in enumerate(ls[5]):
            if i == True:
                left_positions.append(counter)
                left_mistakes += 1

    if get_class_name(rs) == True:
        for counter,i in enumerate(rs[5]):
            if i == False:
                right_positions.append(counter)
                right_mistakes += 1
    else:
        for counter,i in enumerate(rs[5]):
            if i == True:
                right_positions.append(counter)
                right_mistakes += 1
    return split_col, ls, rs, (left_mistakes + right_mistakes), left_positions, right_positions


def create_testing_data(inputfile):
    """
    Preprocessing testing data for testing
    :param inputfile:
    :return:
    """
    testing_file = open(inputfile, 'r')
    average_word_length_list = preprocess_avg_word_length(testing_file)
    testing_file = open(inputfile, 'r')
    article_list = preprocess_articles(testing_file)
    testing_file = open(inputfile, 'r')
    preposition_list = preprocess_prepositions(testing_file)
    testing_file = open(inputfile, 'r')
    punctuation_list = preprocess_punctuations(testing_file)
    testing_file = open(inputfile, 'r')
    vowel_pair_list = preprocess_vowel_pairs(testing_file)

    create_testing_dat(average_word_length_list, article_list, preposition_list, punctuation_list, vowel_pair_list)
    test_data = convert_test_csv()


def main(inputfile, classifier_filename,iftrain):
    training_file = open(inputfile, 'r')
    get_data_from_wiki()
    average_word_length_list = preprocess_avg_word_length(training_file)
    training_file = open(inputfile, 'r')
    article_list = preprocess_articles(training_file)
    training_file = open(inputfile, 'r')
    preposition_list = preprocess_prepositions(training_file)
    training_file = open(inputfile, 'r')
    punctuation_list = preprocess_punctuations(training_file)
    training_file = open(inputfile, 'r')
    vowel_pair_list = preprocess_vowel_pairs(training_file)
    training_file = open(inputfile, 'r')
    target = create_target_column(training_file)
    create_dat(average_word_length_list, article_list, preposition_list, punctuation_list, vowel_pair_list, target)
    train_data = convert_csv()
    if iftrain:
        write_classifier(train_data, "testing_dataset.dat", classifier_filename)
    else:
        create_testing_data(inputfile)

    import classifier as c
    c.main()


if __name__ == '__main__':
    main()
