import array
#import numpy
import datetime
import os
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    cohen_kappa_score,
    confusion_matrix,
    hinge_loss,
    matthews_corrcoef,
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_similarity_score,
    log_loss,
    precision_recall_fscore_support,
    zero_one_loss,
    average_precision_score,
    roc_auc_score,
)

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {}'.format(now))
    print('Time taken: {}'.format(now-last_time))
    return now

def get_wind(line):
    if int(line.split(',')[-1].split('.')[0]) >= 15:
        return 1
    return 0

last_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: {}'.format(last_time))
actual_file = r'.\data\In_situMeasurementforTraining_201712.csv'
#actual_data = array.array('f')
#actual_data = array.array('i')
actual_data = []

# load actual data
print('Loading real data')
with open(actual_file, 'r') as f:
    for line in f:
        actual_data.append(get_wind(line))
last_time = get_time(last_time)

for model in range(1, 11):

    print('\n------------------------------------------')
    print('Loading Model {}'.format(model))

    # load data for model N
    #test_data = array.array('f')
    #test_data = array.array('i')
    test_data = []
    for day in range(1, 6):
        for hour in range(3, 21):
            test_file = r'.\data_parsed\Training_Day{}_Hour{}_Model{}.csv'.format(day, hour, model)
            with open(test_file, 'r') as f:
                for line in f:
                    test_data.append(get_wind(line))
    last_time = get_time(last_time)

    # compare and stuff

    print('\nPrecision Recall Curve')
    print(precision_recall_curve(actual_data, test_data))
    print('\nROC Curve')
    print(roc_curve(actual_data, test_data))
    print('\nCohen Kappa Score')
    print(cohen_kappa_score(actual_data, test_data))
    print('\nConfusion Matrix')
    print(confusion_matrix(actual_data, test_data))
    print('\nHinge Loss')
    print(hinge_loss(actual_data, test_data))
    print('\nMatthews Correlation Coefficient')
    print(matthews_corrcoef(actual_data, test_data))
    print('\nAccuracy Score')
    print(accuracy_score(actual_data, test_data))
    print('\nClassification Report')
    print(classification_report(actual_data, test_data))
    print('\nF1 Score')
    print(f1_score(actual_data, test_data))
    print('\nHamming Loss')
    print(hamming_loss(actual_data, test_data))
    print('\nJaccard Similarity Score')
    print(jaccard_similarity_score(actual_data, test_data))
    # causes memory errors
    # print('\nLog Loss')
    # print(log_loss(actual_data, test_data))
    print('\nPrecision Recall F-Score Support')
    print(precision_recall_fscore_support(actual_data, test_data))
    print('\nZero-One Loss')
    print(zero_one_loss(actual_data, test_data))
    print('\nAverage Precision Score')
    print(average_precision_score(actual_data, test_data))
    print('\nROC AUC Score')
    print(roc_auc_score(actual_data, test_data))

    print('\n')
    last_time = get_time(last_time)    
