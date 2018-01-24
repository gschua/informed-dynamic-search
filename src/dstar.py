import array
import heapq
import time
import os


class AStar:

    '''Subset of Graph, such that it is a graph whose edges are only with its
    4-adjacent neighbors.'''

    def __init__(self, xsize, ysize, start, goal, max_steps, debug=False):

        self.xsize = xsize
        self.ysize = ysize
        self.start_tuple = start
        self.goal_tuple = goal
        self.max_steps_per_map = max_steps
        self.DEBUG = debug

        self.max_steps_total = 0
        self.node_count = self.xsize * self.ysize
        self.start_index = self.tuple_to_array(start[0], start[1])
        self.goal_index = self.tuple_to_array(goal[0], goal[1])

        self.nodes = [{'parent': -1, 'cost': -1, 'steps': 0, 'visited': False} for i in range(self.node_count)]
        '''
        Cost: Probabilty of wind speed >= 15, range is [0, 100]
        Steps: Number of nodes from start to current
        Status:
            0: NEW, meaning it has never been placed on the OPEN list
            1: OPEN, meaning it is currently on the OPEN list
            2: CLOSED, meaning it is no longer on the OPEN list
            3: RAISE, indicating its cost is higher than the last time it was on the OPEN list
            4: LOWER, indicating its cost is lower than the last time it was on the OPEN list
        '''

        self.nodes[self.start_index]['cost'] = 0
        self.nodes[self.start_index]['visited'] = True
        self.frontier = [(0, self.start_index)]   # (priority, node)
        self.next_frontier = []
        heapq.heapify(self.frontier)

    def set_graph(self, graph):
        self.graph = graph
        self.heuristic_constant = self.get_heuristic_constant()

    def get_heuristic_constant(self):
        sum = 0
        for g in self.graph:
            if g >= 0:
                sum += g
        return int(sum/self.node_count)

    def tuple_to_array(self, xcoor, ycoor):
        return ycoor * self.xsize + xcoor

    def array_to_tuple(self, array_idx):
        return array_idx % self.xsize, int(array_idx/self.xsize)

    def get_neighbors(self, main):

        def north_exists():
            try:
                return north > -1 and self.graph[north] > -1
            except:
                return False
        def south_exists():
            try:
                return south < self.node_count and self.graph[south] > -1
            except:
                return False
        def east_exists():
            try:
                return east < self.node_count and main % self.xsize != self.xsize - 1 and self.graph[east] > -1
            except:
                return False
        def west_exists():
            try:
                return west > -1 and main % self.xsize != 0 and self.graph[west] > -1
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

    def a_star_search(self):

        while self.frontier:

            priority, current = heapq.heappop(self.frontier)
            self.nodes[current]['visited'] = True

            if current == self.goal_index:
                break
            if self.nodes[current]['steps'] == self.max_steps_total:
                continue

            for next in self.get_neighbors(current):

                new_cost = self.nodes[current]['cost'] + self.get_cost(current, next)

                if self.nodes[next]['cost'] < 0 or self.nodes[next]['cost'] > new_cost:

                    self.nodes[next]['cost'] = new_cost
                    self.nodes[next]['parent'] = current
                    self.nodes[next]['steps'] = self.nodes[current]['steps'] + 1
                    priority = new_cost + self.heuristic(next)

                    for f in self.frontier:
                        if f[1] == next:
                            f = (priority, next)
                            heapq.heapify(self.frontier)
                            break
                    else:
                        heapq.heappush(self.frontier, (priority, next))

            #self.illustrate()

        if self.DEBUG:
            print(self.frontier)
            # for i in range(self.node_count):
                # if self.nodes[i]['parent']:
                    # print(i, self.nodes[i])
            self.illustrate()

        # if cost is -1, goal was not reached
        return self.final_path(), self.nodes[self.goal_index]['cost']

    def final_path(self):

        path = []
        idx = self.goal_index
        while idx >= 0 and idx != self.start_index:
            path.append(idx)
            idx = self.nodes[idx]['parent']
        return path

    def illustrate(self):

        row = 0
        for i in range(self.node_count):
            if i == self.start_index:
                print('o', end='')
            elif i == self.goal_index:
                print('+', end='')
            elif self.nodes[i]['visited']:
                print('x', end='')
            elif self.graph[i] < 0:
                print('~', end='')
            else:
                print('.', end='')
            row += 1
            if row >= self.xsize:
                row = 0
                print()

        if self.DEBUG:
            for i in self.frontier:
                print(i)
            time.sleep(2)
        else:
            time.sleep(0.2)

        os.system('cls')


class DStar(AStar):

    def __init__(self, graph_set, *args, **kwargs):

        super(DStar, self).__init__(*args, **kwargs)
        self.graph_set = graph_set

    def d_star_search(self):

        for i, ggg in enumerate(self.graph_set):

            print('Map Change to {}'.format(i + 3))
            self.set_graph(ggg)
            self.max_steps_total += self.max_steps_per_map

            for i in range(self.node_count):
                if self.nodes[i]['visited'] and self.graph[i] >= 0:
                    priority = self.nodes[i]['cost'] + self.heuristic(i)
                    heapq.heappush(self.frontier, (priority, i))

            path, cost = self.a_star_search()

            if cost >= 0:
                break

        return path, cost
