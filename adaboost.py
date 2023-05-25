from random import random
import sys
import decisiontree as l
import math
import pandas as pd

VALUE_E = 2.71828182


def update_df_with_wt(all_data):
    """
    UpUpdating Dataframe with weights
    :param all_data:
    :return:
    """
    updated_wt = 1 / len(all_data)
    all_data.insert(6, "UW", [updated_wt for i in range(len(all_data))], True)
    return all_data


def get_weights_sum(data, positions):
    """
    Getting the sum of the weights
    :param data:
    :param positions:
    :return:
    """
    sum = 0
    for i in positions:
        sum += float(data.iloc[i]["UW"])
    return sum


def update_b4_nor(data, amount_of_say, l_positions):
    """
    Step before normalization using the appropriate formula
    :param data:
    :param amount_of_say:
    :param l_positions:
    :return:
    """
    weight_list = []
    for count, i in enumerate(data["UW"]):
        if count in l_positions:
            weight_list.append(data.iloc[count]["UW"] * math.pow(VALUE_E, amount_of_say))
        else:
            weight_list.append(data.iloc[count]["UW"] * math.pow(VALUE_E, - amount_of_say))
    data = data.drop("UW", axis=1)
    data.insert(6, 'UW', weight_list)
    return data


def stepOne(all_data):
    """
    Using the correct formulas for the first step
    :param all_data:
    :return:
    """
    ec = []
    say_dict = {}
    split_cols = []
    while len(ec) < 5:
        split_col, ls, rs, total_mistakes, l_positions, r_positions = l.dt_helper(all_data, ec)
        split_cols.append(split_col)
        left_weights_sum = get_weights_sum(ls, l_positions)
        right_weights_sum = get_weights_sum(rs, r_positions)
        total_weights_sum = left_weights_sum + right_weights_sum
        if total_weights_sum == 0:
            break
        amount_of_say = 0.5 * math.log(((1 - total_weights_sum) / total_weights_sum), 2)
        say_dict[split_col] = amount_of_say
        left_updated_weights = update_b4_nor(ls, amount_of_say, l_positions)
        right_updated_weights = update_b4_nor(rs, amount_of_say, r_positions)
        join_left_right = pd.concat([left_updated_weights, right_updated_weights])
        normalized_data = normalize(join_left_right, "UW")
        all_data = cumilative_data(normalized_data, "UW")
        ec.append(split_col)
        all_data.index = pd.RangeIndex(len(all_data.index))
        all_data.index = range(len(all_data.index))
    return say_dict


def normalize(all_data, column):
    """
    Normalizing the column weights
    :param all_data:
    :param column:
    :return:
    """
    suma = all_data[column].sum()
    all_data[column] = (all_data[column] / suma)
    return all_data


def cumilative_data(all_data, colum):
    """
    Method to select the larger nearest number
    :param all_data:
    :param colum:
    :return:
    """
    new_df = []

    cumilative = all_data[colum].cumsum()
    suma = 0
    extra = []
    for i in range(len(cumilative) - 1):
        extra.append(suma + cumilative[i])
    for i in range(len(cumilative)):
        ran = random()
        position = closest(cumilative, extra, ran)
        new_df.append(all_data.iloc[position])
    df = pd.DataFrame(new_df)
    return df


def closest(lst, extra, K):
    """
    Calculate closest greater number in the column
    :param lst:
    :param extra:
    :param K:
    :return:
    """
    extra = [0]
    suma = 0
    for i in range(len(lst) - 1):
        extra.append(suma + lst[i])
    for i in range(len(lst)):
        if extra[i] < K < lst[i]:
            return i


def write_classifier(train_data, file_name, says, classifier_filename):
    """
    Writes classifier file
    :param train_data:
    :param file_name:
    :param says:
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
def classify(data_list, says):
\twith open('ada_predictions.csv', 'w') as f:
\t\tfo = csv.writer(f)
\t\tfor i in range(len(data_list)):
\t\t\teng_sum = 0
\t\t\tdutch_sum = 0
\t\t\tfor column, say in says.items():
\t\t\t\tif data_list.iloc[i][column] == 1:
\t\t\t\t\tdutch_sum += say
\t\t\t\telse:
\t\t\t\t\teng_sum += say
\t\t\tif dutch_sum > eng_sum:
\t\t\t\tfo.writerow(['en'])
\t\t\t\tprint("en")
\t\t\telse:
\t\t\t\tfo.writerow(['nl'])
\t\t\t\tprint("nl")
\t\tf.close()\n\n
def main(traindata):
\tdecisiontree.main({},"",False)
\tdata = create_pd('testing_dataset.dat')
\tclassify(data,{})
\n\n
if __name__ == \'__main__\':
\tmain()'''.format("traindata", says))
    f.close()


def main(inputfile, classifier_filename):
    l.main(inputfile, classifier_filename, True)
    training_file = open(inputfile, 'r')
    l.get_data_from_wiki()
    average_word_length_list = l.preprocess_avg_word_length(training_file)
    training_file = open(inputfile, 'r')
    article_list = l.preprocess_articles(training_file)
    training_file = open(inputfile, 'r')
    preposition_list = l.preprocess_prepositions(training_file)
    training_file = open(inputfile, 'r')
    punctuation_list = l.preprocess_punctuations(training_file)
    training_file = open(inputfile, 'r')
    vowel_pair_list = l.preprocess_vowel_pairs(training_file)
    training_file = open(inputfile, 'r')
    target = l.create_target_column(training_file)
    l.create_dat(average_word_length_list, article_list, preposition_list, punctuation_list, vowel_pair_list, target)
    train_data = l.convert_csv()
    # test_data = l.create_testing_data()
    allData = update_df_with_wt(train_data)
    says = stepOne(allData[:500])
    write_classifier(train_data, "testing_dataset.dat", says, classifier_filename)
    import classifier2 as c
    c.main()


if __name__ == '__main__':
    main()
