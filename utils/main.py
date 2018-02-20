import array
import datetime
import sys
import pdb
from dstar import DStar
from init import *

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {} (Took {})'.format(now, now-last_time))
    return now

def get_wind(line):
    return int(line.split(',')[-1])

def get_graph(day):

    print('\n---------------------------------\n')
    print('Loading {} graph data (Day {})'.format(data_type, day))

    data = [array.array('i', [0 for j in range(size)]) for i in range(last_hour-first_hour)]
    #data = [[0.0 for j in range(size)] for i in range(last_hour-first_hour)]

    for hour in range(first_hour, last_hour):
        with open(get_file(day, hour), 'r') as f:
            idx = 0
            for line in f:
                data[hour-first_hour][idx] = get_wind(line)
                idx += 1

    return data

if __name__ == '__main__':

    day = int(sys.argv[2])
    last_time = datetime.datetime.now().replace(microsecond=0)
    print('Time start: {}'.format(last_time))

    graph_set = get_graph(day)
    last_time = get_time(last_time)

    for city, goal in cities.items():

        print('\n---------------------------------\n')
        print('Day {}, City {}'.format(day, city))

        ds = DStar(
            graph_set,
            max_cost,
            dstar_path_output_format.format(d=str(day).zfill(2), c=str(city).zfill(2)),
            cost_threshold_list,
            map_limits,
            graph_num_start,
            xsize,
            ysize,
            start,
            goal,
            zero_index,
            limit_type,
            dstar_graph_output_format.format(d=str(day).zfill(2), c=str(city).zfill(2)),
        )
        ds.dstar_search()
        last_time = get_time(last_time)
