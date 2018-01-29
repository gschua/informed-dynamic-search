from dstar import DStar
import datetime
import array
import os

def get_wind(line):
    if int(line.split(',')[-1].split('.')[0]) >= 15:
        return -1
    return 1

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {} (Took {})'.format(now, now-last_time))
    return now

start_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: {}'.format(start_time))

graph_set = [array.array('i') for i in range(21-3)]
graph_file = './real_data/Day{}_Hour{}.csv'

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

for day in range(1, 6):

    # load graph
    print('Loading graph data (Day {})'.format(day))
    graph_set = [array.array('i') for i in range(21-3)]
    for hour in range(3, 21):
        with open(graph_file.format(day, str(hour).zfill(2)), 'r') as f:
            for line in f:
                graph_set[hour-3].append(get_wind(line))
    last_time = get_time(start_time)

    for city_num, city_coor in cities.items():

        print('City {}'.format(city_num))
        ds = DStar(graph_set, 548, 421, (142, 328), city_coor, 30, 'City{}_Day{}'.format(city_num, day), debug=True)
        path, cost = ds.d_star_search()

        if cost < 0:
            print('Path not found for City {} on Day {}'.format(city_num, day))
            continue

        full_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'City{}_Day{}_Final.txt'.format(city_num, day),
        )
        with open(full_file_path, 'w') as f:
            f.write('Cost: {}\n'.format(cost))
            f.write('coordinates,cost,first_map,max_steps,parent,steps,visited\n')
            for a, b in path:
                f.write('{},'.format(a))
                for k in sorted(b.keys()):
                    f.write('{},'.format(b[k]))
                f.write('\n')
        last_time = get_time(last_time)
