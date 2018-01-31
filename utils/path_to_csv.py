import os
import re

input_location = './results'
output_file = './results.csv'

def get_time(hour, minute):
    return str(hour).zfill(2) + ':' + str(minute).zfill(2)

def path_to_csv(output_file, day, city, path_indices):

    # line format: city,date,time,x-coordinate,y-coordinate
    csv_line = '{c},{d},{t},{x},{y}'
    hour = 3
    minute = 0
    formatted_data = []

    for p in range(len(path_indices)):

        times = []

        if p < len(path_indices) - 1:
            while hour < path_indices[p+1][2]:
                while minute < 60:
                    times.append(get_time(hour, minute))
                    minute += 2
                hour += 1
                minute = 0
            assert hour >= path_indices[p+1][2], '{} {} {} {} {} {} {}'.format(city, day, hour, minute, path_indices[p][0], path_indices[p][1], path_indices[p][2])

        assert minute < 60, '{} {} {} {} {} {} {}'.format(city, day, hour, minute, path_indices[p][0], path_indices[p][1], path_indices[p][2])
        times.append(get_time(hour, minute))
        minute += 2
        if minute >= 60:
            hour += 1
            minute = 0

        for t in times:
            formatted_data.append(csv_line.format(c=city, d=day, t=t, x=path_indices[p][0], y=path_indices[p][1]))

    with open(output_file, 'a') as f:
        for d in formatted_data:
            f.write(d)
            f.write('\n')

def get_id(input_file):
    day = re.search('Day(\d+?)', input_file).group(1)
    city = re.search('City(\d+?)', input_file).group(1)
    return int(day)+5, int(city)

def get_path(input_file):
    # line format: index,xcoor,ycoor,map,node_cost,total_cost,parent,steps
    # we want xcoor, ycoor, and map
    path_indices = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                x = line.split(',')
                path_indices.append((x[1], x[2], int(x[3])))
            except:
                continue
    # reverse so start is first and goal is last
    path_indices.reverse()
    return path_indices

if __name__ == '__main__':
    for dir, subdirs, files in os.walk(input_location):
        for f in sorted(files):
            if 'PATH' in f:
                input_file = os.path.join(dir, f)
                day, city = get_id(input_file)
                path_indices = get_path(input_file)
                path_to_csv(output_file, day, city, path_indices)
