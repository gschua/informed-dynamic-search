'''util to split data into easier to parse chunks'''
import os
import sys

def write_to_file(data, file_count):
    global new_file_name
    global num_fill
    nfn = new_file_name.format(str(file_count).zfill(num_fill))
    with open(nfn, 'w') as g:
        print(nfn)
        g.write(''.join(data))

file_path_long = sys.argv[1]
line_count = int(sys.argv[2])
num_fill = int(sys.argv[3])
assert os.path.isfile(file_path_long)

file_path_short, file_ext = os.path.splitext(file_path_long)
new_file_name = file_path_short + '_{}' + file_ext
file_count = 1
data = []

with open(file_path_long, 'r') as f:
    for line in f:
        data.append(line)
        if len(data) >= line_count:
            write_to_file(data, file_count)
            data = []
            file_count += 1
if data:
    write_to_file(data, file_count)