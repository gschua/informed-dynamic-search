'''util to split test and training data by model into smaller csv files'''
import datetime
import os
import sys

def write_to_file(line, file):
    with open(file, 'ab') as g:
        g.write(line)

start_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: %s' % start_time)

input_file_long = sys.argv[1]
assert os.path.isfile(input_file_long)

input_file_short, file_ext = os.path.splitext(input_file_long)
count = 0
new_files = []

for i in range(1, 11):
    new_files.append(input_file_short + '_model{}'.format(str(i).zfill(2)) + file_ext)

with open(input_file_long, 'rb') as f:
    f.seek(32, 1)    # skip the first row
    for line in f:
        write_to_file(line, new_files[count % 10])
        count += 1

end_time = datetime.datetime.now().replace(microsecond=0)
print('Count: %s' % count)
print('Time end: %s' % end_time)
print('Time taken: %s' % (end_time - start_time))