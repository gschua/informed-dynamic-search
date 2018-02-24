import array
import datetime
import sys
from dstar import DStar
from init import *

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {} (Took {})'.format(now, now-last_time))
    return now

def get_graph(day):
    print('\n---------------------------------\n')
    print('Loading {} graph data (Day {})'.format(data_type, day))
    data = [array.array('i', [0 for j in range(size)]) for i in range(first_hour, last_hour)]
    for hour in range(last_hour-first_hour):
        with open(get_file(day, hour), 'r') as f:
            for i, line in enumerate(f):
                data[hour][i] = int(line.split(',')[-1])
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
            xsize,
            ysize,
            start,
            goal,
            zero_index,
            heuristic_type,
            limit_type,
            dstar_graph_output_format.format(d=str(day).zfill(2), c=str(city).zfill(2)),
        )
        ds.dynamic_search()
        last_time = get_time(last_time)
