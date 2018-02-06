'''Splitter for Met Weather Data

Separates data into smaller csv files by date and model for easier parsing by
other scripts. In the case of real data, it is only separated by date.
Assumes that the data is grouped by date, then location, then model.
Assumes that there is a header. The subfiles will not have a header.

e.g. 
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

Expects 2 arguments:
1. data type ('real', 'train', or 'test')
2. name (NOT path) of input file
'''
import copy
import datetime
import os
import sys

###############################################################################
# Const and Default declarations

data_types = ['real', 'train', 'test']
root_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(root_dir, 'data')
output_path = os.path.join(root_dir, 'data_parsed')

first_day = 1
first_hour = 3
last_hour = 21
model_count = 10

###############################################################################
# Init

data_type = sys.argv[1]
input_file_name = sys.argv[2]
input_file = os.path.join(input_path, input_file_name)
assert os.path.isfile(input_file), '{} does not exist'.format(input_file)
assert os.path.isdir(output_path), '{} does not exist'.format(output_path)
assert data_type in data_types, 'Invalid data type: {}'.format(data_type)

# to avoid writing another script, we treat real data as though it has 1 model
if data_type == 'real':
    model_count = 1

if data_type == 'test':
    first_day = 6

size = 548 * 421 * model_count
output_format = data_type + '_d{d}h{h}m{m}.csv'
data = [[] for i in range(model_count)]

###############################################################################
# Begin Splitting

start_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: %s' % start_time)

count = 0
day = copy.deepcopy(first_day)
hour = copy.deepcopy(first_hour)

with open(input_file, 'r') as f:

    for line in f:

        count += 1
        # ignore header
        if count == 1:
            continue

        # -2 because of header, change to -1 if input file has no header
        current_model = (count-2) % model_count
        data[current_model].append(line)

        if count % size == 1:

            # at this point we have gathered all the data for a particular
            # day-hour combo, so we write them all to files
            for model in range(model_count):
                output_file = os.path.join(output_path, output_format.format(
                    d=str(day).zfill(2),
                    h=str(hour).zfill(2),
                    m=str(model+1).zfill(2)
                ))
                with open(output_file, 'w') as g:
                    for d in data[model]:
                        g.write(d)

            # reset or increment variables
            data = [[] for i in range(model_count)]
            hour += 1
            if hour == last_hour:
                hour = copy.deepcopy(first_hour)
                day += 1

end_time = datetime.datetime.now().replace(microsecond=0)
print('Count: %s' % count)
print('Time end: %s' % end_time)
print('Time taken: %s' % (end_time - start_time))
