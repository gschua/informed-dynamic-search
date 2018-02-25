import os
from init import *

def parse_path(path_file):
    path = []
    with open(path_file, 'r') as f:
        for i, line in enumerate(f):
            # skip the headers
            if i < 3:
                continue
            path.append(line.strip().split(','))
    return path

def get_path_file(day, city):
    file_prefix = 'd{d}c{c}'.format(
        d=str(day).zfill(2),
        c=str(city).zfill(2),
    )
    dir = os.path.join(root_dir, 'results', 'real')
    for filename in os.listdir(dir):
        if filename.startswith(file_prefix) and 'path' in filename and not 'graph' in filename:
            return os.path.join(dir, filename)
    return None

def get_actual_data(day):
    data = []
    for hour in range(18):
        data.append([])
        with open(get_file(day, hour), 'r') as f:
            for line in f:
                data[hour].append(True if int(line.split(',')[-1]) > 10 else False)
    return data

def get_score(graph, path):

    score = 0
    map = 0
    prev_map = 0
    steps = 0
    max_steps = 30

    for node in path:

        map = int(node[3])
        if graph[map][int(node[0])]:
            score = 'CRASH'
            break

        if prev_map != map:
            if steps >= max_steps:
                raise Exception
            while steps < max_steps:
                steps += 1
                score += 2
            steps = 0

        score += 2
        prev_map = map

    return score

if __name__ == '__main__':

    for day in range(1, 6):

        graph_set = get_actual_data(day)

        for city in range(1, 11):

            path_file = get_path_file(day, city)
            if not path_file:
                print('No path for Day {} City {}'.format(day, city))
                continue
            path = parse_path(path_file)
            score = get_score(graph_set, path)
            print('Day {} City {} Score: {}'.format(day, city, score))
