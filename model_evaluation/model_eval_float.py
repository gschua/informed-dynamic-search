import array
#import numpy
import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
)

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {}'.format(now))
    print('Time taken: {}'.format(now-last_time))
    return now

def get_wind(line):
    l = line.split(',')[-1].strip('\n')
    return(float(l))
    try:
        if len(l.split('.')[1]) == 1:
            l += '0'
        l = l.replace('.', '', 1)
    except IndexError:
        l += '00'
    return int(l)

# init
start_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: {}'.format(start_time))
test_file = r'data\ForecastDataforTraining_201712.csv'
actual_file = r'data\In_situMeasurementforTraining_201712.csv'
model_number = 9
test_data = array.array('f')
actual_data = array.array('f')
#test_data = array.array('i')
#actual_data = array.array('i')
#test_data = []
#actual_data = []

# load actual data
count = 0
with open(actual_file, 'r') as f:
    for line in f:
        try:
            actual_data.append(get_wind(line))
        except ValueError as e:
            pass
        count += 1
print('Actual Data Count: {}'.format(count))
last_time = get_time(last_time)

# load data for model N
for model_number in range(10):

    print('Model {}'.format(model_number))
    count = 0
    with open(test_file, 'r') as f:
        for line in f:
            if count % 10 == model_number:
                try:
                    test_data.append(get_wind(line))
                except ValueError as e:
                    pass
            count += 1
            # if count == 9 * 10 + 1:
                # break
    print('Test Data Count: {}'.format(count))
    last_time = get_time(start_time)

    # convert to numpy array, https://stackoverflow.com/a/5675147
    # np_test_data = []
    # for td in test_data:
        # #np_test_data.append(numpy.frombuffer(td, dtype=numpy.float16))
        # np_test_data.append(list(td))
    # #np_actual_data = numpy.frombuffer(actual_data, dtype=numpy.float16)
    # np_actual_data = list(actual_data)

    # compare and stuff

    print('\nMean Squared Error')
    print(mean_squared_error(actual_data, test_data))
    last_time = get_time(last_time)

    print('\nMean Absolute Score')
    print(mean_absolute_error(actual_data, test_data))
    last_time = get_time(last_time)

    print('\nExplained Variance Score')
    print(explained_variance_score(actual_data, test_data))
    last_time = get_time(last_time)

    print('\nR2 Score')
    print(r2_score(actual_data, test_data))
    last_time = get_time(last_time)
