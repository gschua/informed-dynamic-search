from dstar import DStar
import datetime
import array
import os
import sys

def get_real_graph(day):

    print('\n---------------------------------\n')
    print('Loading real graph data (Day {})'.format(day))

    graph_file = './real/Real_Day{}_Hour{}.csv'
    graph_set = [array.array('i') for i in range(21-3)]

    for hour in range(3, 21):
        with open(graph_file.format(day, str(hour).zfill(2)), 'r') as f:
            for line in f:
                graph_set[hour-3].append(get_wind_1(line))

    return graph_set

def get_predicted_graph(day, typ):

    print('\n---------------------------------\n')
    print('Loading {} graph data (Day {})'.format(typ, day))

    graph_file = './{0}/{0}_Day{1}_Hour{2}_Model{3}.csv'
    graph_set = [array.array('i', [0 for j in range(548*421)]) for i in range(21-3)]
    #graph_set = [[0.0 for j in range(548*421)] for i in range(21-3)]

    for hour in range(3, 21):
        for model in range(1, 11):
            with open(graph_file.format(typ, day, hour, model), 'r') as f:
                idx = 0
                for line in f:
                    graph_set[hour-3][idx] += get_wind(line)
                    idx += 1

    return graph_set

def get_wind(line):
    if int(line.split(',')[-1].split('.')[0]) >= 15:
        return 1
    return 0

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {} (Took {})'.format(now, now-last_time))
    return now

if __name__ == '__main__':

    #typ = 'real'
    #typ = 'training'
    typ = 'testing'
    xsize = 548
    ysize = 421
    start_tuple = (142, 328)
    max_steps_per_map = 30
    #max_cost = 1
    max_cost = 10
    threshold_increment = 3
    dump_path = r'./results/{}_City{}_Day{}'
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
    last_time = datetime.datetime.now().replace(microsecond=0)
    print('Time start: {}'.format(last_time))

    #for day in range(2, 6):
    day = int(sys.argv[1])

    graph_set = get_predicted_graph(day, typ)
    last_time = get_time(last_time)

    for city_num, city_coor in cities.items():

        print('\n---------------------------------\n')
        print('Day {}, City {}'.format(day, city_num))

        ds = DStar(
            graph_set,
            max_cost,
            threshold_increment,
            xsize,
            ysize,
            start_tuple,
            city_coor,
            max_steps_per_map,
            dump_path.format(typ, city_num, day),
            debug=True
        )
        ds.d_star_search()
        last_time = get_time(last_time)
