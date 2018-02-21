'''Parser for Met Weather Data

Separates data into smaller csv files by date and model for easier parsing by
other scripts. In the case of real data, it is only separated by date.
Assumes that the data is grouped by date, then location, then model.
Assumes that there is a header. The subfiles will not have a header.
Changes the sorting of the locations from y than x to a x than y to maintain
comptability with DStar.

Example:

Date | Location | Model | Value
1, 3 | 1, 1     | 1     | 999
1, 3 | 1, 1     | 2     | 888
...
1, 3 | 1, 1     | 9     | 777
1, 3 | 1, 2     | 1     | 666
1, 3 | 1, 2     | 2     | 555
...
1, 3 | 99, 99   | 9     | 444
1, 4 | 1, 1     | 1     | 999
...

Becomes:

file_1_3.csv
Location | Value
1, 1     | 999
2, 1     | 888
3, 1     | 777
...
99, 99   | 666
...


Expects 2 arguments:
1. data type ('real', 'train', or 'test')
2. name (NOT path) of input file
'''

import copy
import datetime
import sys
from init import *

def wind_score(wind_speeds):

    '''Put scoring algorithm here'''
    score = 0
    for wind in wind_speeds:
        if int(wind.split('.')[0]) >= 15:
            score += 10
        else:
            score += 1
    return score


def parse_graph(data, day, hour):

    # at this point, "data" has all the info for a particular day-hour combo
    with open(get_file(day, hour), 'w') as f:
        for coordinates in sorted(data.keys(), key=lambda x: (x[1], x[0])):
            f.write('{x},{y},{v}\n'.format(
                x=coordinates[0],
                y=coordinates[1],
                v=wind_score(data[coordinates])
            ))

def parse_raw():

    #data = [[] for i in range(model_count)]
    data = {}
    count = 0
    day = copy.deepcopy(first_day)
    hour = 0

    with open(raw_file, 'r') as f:

        for line in f:

            count += 1
            # ignore header
            if count == 1:
                continue

            # -2 because of header, change to -1 if input file has no header
            # current_model = (count-2) % model_count

            a = line.split(',')
            # convert x and y coordinates to int for numerical sorting
            coordinates = (int(a[0]), int(a[1]))
            wind = a[5]
            # we wont get model because they will be appended to the list
            # in order anyway
           
            if coordinates in data:
                data[coordinates].append(wind)
            else:
                data[coordinates] = [wind]

            if count % (size * model_count) == 1:
                parse_graph(data, day, hour)
                # reset or increment variables
                #data = [[] for i in range(model_count)]
                data = {}
                hour += 1
                if hour == (last_hour-first_hour):
                    hour = 0
                    day += 1


if __name__ == '__main__':

    input_file_name = sys.argv[2]
    raw_file = os.path.join(root_dir, 'data', input_file_name)

    start_time = datetime.datetime.now().replace(microsecond=0)
    print('Time start: %s' % start_time)

    parse_raw()

    end_time = datetime.datetime.now().replace(microsecond=0)
    print('Time end: %s' % end_time)
    print('Time taken: %s' % (end_time - start_time))
