from dstar import AStar, DStar
import datetime
import array

def get_wind(line):
    if int(line.split(',')[-1].split('.')[0]) >= 15:
        return -1
    return 1

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {}'.format(now))
    print('Time taken: {}'.format(now-last_time))
    return now

start_time = datetime.datetime.now().replace(microsecond=0)
print('Time start: {}'.format(start_time))

graph_set = [array.array('i') for i in range(18)]
graph_file = './aaa.csv'

# load graph
count = 0
grid_size = 548 * 421
map = 0

print('Loading graph data')
with open(graph_file, 'r') as f:
    for line in f:
        if count == grid_size:
            map += 1
            count = 0
        if map >= 18:
            break
        graph_set[map].append(get_wind(line))
        count += 1
last_time = get_time(start_time)

print('Performing bad D* Search')
ds = DStar(graph_set, 548, 421, (142, 328), (189,274), 30, debug=True)
path, cost = ds.d_star_search()
print(path)
print(cost)
last_time = get_time(last_time)

# 1,84,203
# 2,199,371
# 3,140,234
# 4,236,241
# 5,315,281
# 6,358,207
# 7,363,237
# 8,423,266
# 9,125,375
# 10,189,274