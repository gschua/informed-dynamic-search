from dstar import AStar
from init import *

###############################################################################
# Simple tests

city_coor = (84, 203)

###############################################################################
# Test tuple <-> index conversion

def test1():

    def zzz(aass, a):
        try:
            print()
            print(a)
            x = aass.tuple_to_index(*a)
            print(x)
            y = aass.index_to_tuple(x)
            print(y)
        except Exception as e:
            print(e)

    aass = AStar(xsize, ysize, start_tuple, city_coor, zero_index)
    AAA = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 10),
        (142, 328),
        (141, 327),
        (548, 421),
        (547, 421),
    ]

    # Test 1
    for a in AAA:
        zzz(aass, a)

    # Test 2
    my_neighbors = aass.get_neighbors(aass.tuple_to_index(*start_tuple))
    for m in my_neighbors:
        print(m, aass.index_to_tuple(m))

    # Test 3
    for a in [(84, 203), (85, 203), (84, 202), (0, 0)]:
        print(abs(city_coor[0] - a[0]) + abs(city_coor[1] - a[1]))

if 1:
    zero_index = True
    test1()
    zero_index = False
    test1()

###############################################################################
# Test 4-adjacent distance calculation

def test2():

    def distance1(ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)

    def distance2(a, b):
        c = abs(a-b)
        d = c % aass.xsize + 1
        return int(c/aass.xsize) + min(d, aass.xsize-d)

    aass = AStar(xsize, ysize, start_tuple, city_coor, zero_index)
    my_tuple = (142, 238)
    my_index = aass.tuple_to_index(*my_tuple)
    for xcoor in range(200, 300):
        for ycoor in range(200, 300):
            test_index = aass.tuple_to_index(xcoor, ycoor)
            if distance1(*my_tuple, xcoor, ycoor) != distance2(my_index, test_index):
                print((xcoor, ycoor))
                print(my_tuple)
                print(distance1(*my_tuple, xcoor, ycoor))
                print(test_index)
                print(my_index)
                print(distance2(my_index, test_index))
                return

if 1:
    zero_index = True
    test2()
    zero_index = False
    test2()

###############################################################################
# Test neighbors

def test3():

    aass = AStar(xsize, ysize, start_tuple, city_coor, zero_index)
    AAA = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 10),
        (142, 328),
        (548, 421),
        (547, 421),
    ]

    for a in AAA:
        print()
        print(a)
        for b in aass.get_neighbors(aass.tuple_to_index(*a)):
            print('{} {}'.format(aass.index_to_tuple(b), b))

if 1:
    zero_index = True
    test3()
    zero_index = False
    test3()


###############################################################################
# Intermediate tests

###############################################################################
# Initialize

day = 6
hour = 3

def get_graph():
    data = []
    with open(get_file(day, hour), 'r') as f:
        for line in f:
            data.append(line.split(','))
            data[-1][-1] = int(data[-1][-1])
    return data

raw_graph = get_graph()
graph = [i[-1] for i in raw_graph]

# only parameters that matter for this test are xsize, ysize, zero_index
astar = AStar(xsize, ysize, start, (3, 2), zero_index, limit_type,
    dstar_graph_output_format.format(d=str(day).zfill(2), c='BBBB'))
astar.set_graph(graph)
astar.set_cost_threshold(50)
astar.reset()

###############################################################################
# Test index_to_tuple() and tuple_to_index()

# with open(get_file(day, str(hour) + 'AAAAA'), 'w') as f:
    # for idx, g in enumerate(raw_graph):
        # a = astar.index_to_tuple(idx)
        # b = astar.tuple_to_index(*a)
        # f.write(str((int(g[0]), int(g[1]))) + ';' + str(idx) + ';' + str(a) + ';' + str(b) + '\n')

for idx, g in enumerate(raw_graph):
    a = astar.index_to_tuple(idx)
    b = astar.tuple_to_index(*a)
    if int(g[0]) != a[0] or int(g[1]) != a[1] or idx != b:
        print(g)
        print(a)
        print(idx)
        print(b)
        raise Exception
print('Conversion functions success')

###############################################################################
# Test illustrate()

# play around with goal and start tuples to check locations map correctly
astar.illustrate()

###############################################################################
# Test get_neighbors()

AAA = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 2),
    (2, 10),
    (142, 328),
    (141, 327),
    (548, 421),
    (547, 420),
]

for a in AAA:
    aa = astar.tuple_to_index(*a)
    neighbors = astar.get_neighbors(aa)
    print()
    print(a, aa)
    for n in neighbors:
        print(astar.index_to_tuple(n), n)
