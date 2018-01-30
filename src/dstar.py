import array
import heapq
import time
import os
import datetime

def get_time(last_time):
    now = datetime.datetime.now().replace(microsecond=0)
    print('Time now: {} (Took {})'.format(now, now-last_time))
    return now

class AStar:

    '''Subset of Graph, such that it is a graph whose edges are only with its
    4-adjacent neighbors.'''

    def __init__(self, xsize, ysize, start, goal, max_steps, file_prefix, debug=False):

        self.xsize = xsize
        self.ysize = ysize
        self.start_tuple = start
        self.goal_tuple = goal
        self.max_steps_per_map = max_steps
        self.file_prefix = file_prefix
        self.DEBUG = debug

        self.node_count = self.xsize * self.ysize
        self.start_index = self.tuple_to_array(start[0], start[1])
        self.goal_index = self.tuple_to_array(goal[0], goal[1])

    def reset(self):

        self.nodes = [
            {
                'map': -1,
                'max_steps': -1,
                'parent': -1,
                'steps': 0,
                'total_cost': -1,
                'visited': False,
            }
            for i in range(self.node_count)
        ]
        self.nodes[self.start_index]['visited'] = True
        self.nodes[self.start_index]['max_steps'] = self.max_steps_per_map
        self.frontier = [(0, self.start_index)]   # (priority, node)

    def set_graph(self, graph):
        self.graph = graph
        self.heuristic_constant = self.get_heuristic_constant()
        if self.nodes[self.start_index]['total_cost'] < 0:
            self.nodes[self.start_index]['total_cost'] = self.graph[self.start_index]

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_heuristic_constant(self):
        sum = 0
        for g in self.graph:
            sum += g
        return int(sum/self.node_count)

    def tuple_to_array(self, xcoor, ycoor):
        return ycoor * self.xsize + xcoor

    def array_to_tuple(self, array_idx):
        return array_idx % self.xsize, int(array_idx/self.xsize)

    def get_neighbors(self, main):

        def north_exists():
            try:
                return north > -1 and self.graph[north] <= self.threshold
            except:
                return False
        def south_exists():
            try:
                return south < self.node_count and self.graph[south] <= self.threshold
            except:
                return False
        def east_exists():
            try:
                return east < self.node_count and main % self.xsize != self.xsize - 1 and self.graph[east] <= self.threshold
            except:
                return False
        def west_exists():
            try:
                return west > -1 and main % self.xsize != 0 and self.graph[west] <= self.threshold
            except:
                return False

        # hopefully main is an int of range [0, self.node_count)
        # but adding a check wastes time

        neighbors = []

        north = main - self.xsize
        south = main + self.xsize
        east = main + 1
        west = main - 1

        if north_exists():
            neighbors.append(north)
        if south_exists():
            neighbors.append(south)
        if east_exists():
            neighbors.append(east)
        if west_exists():
            neighbors.append(west)

        return neighbors

    def get_cost(self, current, next):
        return self.graph[next]

    def heuristic(self, node):
        xcoor, ycoor = self.array_to_tuple(node)
        return (abs(self.goal_tuple[0] - xcoor) + abs(self.goal_tuple[1] - ycoor)) * self.heuristic_constant

    def a_star_search(self, map_number):

        while self.frontier:

            priority, current = heapq.heappop(self.frontier)
            self.nodes[current]['visited'] = True
            self.nodes[current]['map'] = map_number

            if current == self.goal_index:
                break
            if self.nodes[current]['steps'] >= self.nodes[current]['max_steps']:
                continue

            for next in self.get_neighbors(current):

                new_cost = self.nodes[current]['total_cost'] + self.get_cost(current, next)

                if self.nodes[next]['total_cost'] < 0 or self.nodes[next]['total_cost'] > new_cost:

                    self.nodes[next]['total_cost'] = new_cost
                    self.nodes[next]['parent'] = current
                    self.nodes[next]['max_steps'] = self.nodes[current]['max_steps']
                    self.nodes[next]['steps'] = self.nodes[current]['steps'] + 1
                    priority = new_cost + self.heuristic(next)

                    for f in self.frontier:
                        if f[1] == next:
                            f = (priority, next)
                            heapq.heapify(self.frontier)
                            break
                    else:
                        heapq.heappush(self.frontier, (priority, next))

        self.illustrate(map_number)

        # if total_cost is -1, goal was not reached
        return self.final_path(), self.nodes[self.goal_index]['total_cost']

    def final_path(self):

        path = []
        idx = self.goal_index
        while idx >= 0 and idx != self.start_index:
            path.append((idx, self.nodes[idx]))
            idx = self.nodes[idx]['parent']
        return path

    def illustrate(self, map_number):

        file_path = self.file_prefix + '_Threshold{}_Hour{}.txt'.format(self.threshold, map_number)
        with open(file_path, 'w') as f:
            row = 0
            for i in range(self.node_count):
                if i == self.start_index:
                    f.write('o ')
                elif i == self.goal_index:
                    f.write('+ ')
                elif self.nodes[i]['visited'] and self.graph[i] > self.threshold:
                    f.write('# ')
                elif self.nodes[i]['visited']:
                    f.write('x ')
                elif self.graph[i] > self.threshold:
                    f.write('~ ')
                else:
                    f.write('. ')
                row += 1
                if row >= self.xsize:
                    row = 0
                    f.write('\n')


class DStar(AStar):

    def __init__(self, graph_set, max_cost, threshold_increment, *args, **kwargs):

        super(DStar, self).__init__(*args, **kwargs)
        self.graph_set = graph_set
        self.max_cost = max_cost
        self.threshold_increment = threshold_increment
        self.last_time = datetime.datetime.now().replace(microsecond=0)

    def write_path(self, path, cost):

        file_path = self.file_prefix + '_PATH_T{}.csv'.format(self.threshold)
        print('Path found! Writing path to {}'.format(file_path))

        with open(file_path, 'w') as f:

            f.write('Cost: {}\n'.format(cost))
            f.write('Threshold: {}/{}\n'.format(self.threshold, self.max_cost))
            f.write('index,xcoor,ycoor,map,node_cost,total_cost,parent,steps\n')

            for idx, node_info in path:
                xcoor, ycoor = self.array_to_tuple(idx)
                f.write(','.join((
                    str(idx),
                    str(xcoor),
                    str(ycoor),
                    str(node_info['map']),
                    str(self.graph_set[node_info['map']-3][idx]),
                    str(node_info['total_cost']),
                    str(node_info['parent']),
                    str(node_info['steps']),
                )))
                f.write('\n')

    def _d_star_search(self):
    
        for i, ggg in enumerate(self.graph_set):

            print('Map change to {}'.format(i + 3))
            self.set_graph(ggg)

            for n in range(self.node_count):
                if self.nodes[n]['visited'] and self.graph[n] >= 0:
                    self.nodes[n]['max_steps'] = self.nodes[n]['steps'] + self.max_steps_per_map
                    priority = self.nodes[n]['total_cost'] + self.heuristic(n)
                    heapq.heappush(self.frontier, (priority, n))

            path, cost = self.a_star_search(i + 3)
            self.last_time = get_time(self.last_time)

            if cost >= 0:
                return path, cost

        return path, cost

    def d_star_search(self):

        threshold = self.threshold_increment
        final_try = False

        while threshold < self.max_cost:

            print('Threshold set to {}'.format(threshold))
            self.set_threshold(threshold)
            self.reset()

            path, cost = self._d_star_search()

            if cost >= 0:
                self.write_path(path, cost)

                if not final_try and threshold + self.threshold_increment < self.max_cost:
                    print('Try one more time with a higher threshold')
                    final_try = True
                    threshold += self.threshold_increment
                    self.reset()
                    continue

            if final_try:
                if cost < 0:
                    print('Path not found with higher threshold')
                return

            print('Path not found, retrying with higher threshold')
            threshold += self.threshold_increment

        print('Failed to find path to goal')
