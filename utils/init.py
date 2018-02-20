import os
import sys

def get_file(day, hour, model='consolidated'):
    return parsed_data_format.format(
        d=str(day).zfill(2),
        h=str(hour).zfill(2),
        m=str(model).zfill(2)
    )

data_types = ['real', 'train', 'test']
root_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(root_dir, 'results')

first_day = 1
first_hour = 3
last_hour = 21
model_count = 10

data_type = sys.argv[1]
assert data_type in data_types, 'Invalid data type: {}'.format(data_type)

# to avoid writing another script, we treat real data as though it has 1 model
if data_type == 'real':
#if 1:
    model_count = 1

if data_type == 'test':
    first_day = 6

# d = day, g/h = hour, m = model, t = threshold, c = city
parsed_data_format = os.path.join(root_dir, 'data_parsed', data_type + '_d{d}h{h}.csv')
dstar_graph_output_format = os.path.join(root_dir, 'results', data_type, 'd{d}c{c}t{{t}}h{{g}}_map.csv')
dstar_path_output_format = os.path.join(root_dir, 'results', data_type, 'd{d}c{c}t{{t}}_path.csv')

xsize = 548
ysize = 421
size = xsize * ysize
max_cost = 100
cost_threshold_list = [11, 51, 101]
map_limits = [30 for i in range(last_hour-first_hour)]
limit_type = 'n'
graph_num_start = 3
start = (142, 328)
cities = {
    '1': (84, 203),
    '2': (199, 371),
    '3': (140, 234),
    '4': (236, 241),
    '5': (315, 281),
    '6': (358, 207),
    '7': (363, 237),
    '8': (423, 266),
    '9': (125, 375),
    '10': (189, 274),
}
zero_index = False
