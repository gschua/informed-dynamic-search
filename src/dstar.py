import array
import heapq
import time
import os

class AStar:

    '''Subset of Graph, such that it is a graph whose edges are only with its
    4-adjacent neighbors.'''

    def __init__(self, graph, xsize, ysize, start, goal, debug=False):

        self.graph = graph  # must be array.array('f')
        self.xsize = xsize
        self.ysize = ysize
        self.start_tuple = start
        self.goal_tuple = goal
        self.DEBUG = debug

        self.node_count = self.xsize * self.ysize
        self.start_index = self.tuple_to_array(start[0], start[1])
        self.goal_index = self.tuple_to_array(goal[0], goal[1])

        #self.visited = array.array('b', [False for i in range(self.node_count)])
        self.visited = [False for i in range(self.node_count)]
        self.parent = [None for i in range(self.node_count)]
        self.cost = [-1 for i in range(self.node_count)]

        self.visited[self.start_index] = True
        self.cost[self.start_index] = 0
        self.frontier = [(0, self.start_index)]   # (priority, node)
        heapq.heapify(self.frontier)

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
            return north > -1 and self.graph[north] > -1
        def south_exists():
            return south < self.node_count and self.graph[south] > -1
        def east_exists():
            return east < self.node_count and main % self.xsize != self.xsize - 1 and self.graph[east] > -1
        def west_exists():
            return west > -1 and main % self.xsize != 0 and self.graph[west] > -1

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

    def search(self):

        while self.frontier:

            priority, current = heapq.heappop(self.frontier)
            if current == self.goal_index:
                break

            for next in self.get_neighbors(current):

                if self.visited[next]:
                    continue

                new_cost = self.cost[current] + self.get_cost(current, next)
                if self.cost[next] < 0 or self.cost[next] > new_cost:
                    self.cost[next] = new_cost
                    self.parent[next] = current
                    priority = new_cost + self.heuristic(next)
                    for node in self.frontier:
                        if node[1] == next:
                            node[0] = priority
                            heapq.heapify(self.frontier)
                            break
                    else:
                        heapq.heappush(self.frontier, (priority, next))

            self.visited[current] = True
            #self.illustrate()

        if self.DEBUG:
            print(self.frontier)
            for i in range(self.node_count):
                if self.parent[i]:
                    print(i, self.parent[i], self.cost[i])
            self.illustrate()

        return self.final_path(), self.cost[self.goal_index]

    def final_path(self):

        path = []
        node = self.goal_index
        while node and node != self.start_index:
            path.append(node)
            node = self.parent[node]
        return path

    def illustrate(self):

        row = 0
        for i, v in enumerate(self.visited):
            if i == self.start_index:
                print('o', end='')
            elif i == self.goal_index:
                print('+', end='')
            elif v:
                print('x', end='')
            elif self.graph[i] < 0:
                print('~', end='')
            else:
                print('.', end='')
            row += 1
            if row >= self.xsize:
                row = 0
                print()

        # if self.DEBUG:
            # # for i in self.frontier:
                # # print(i)
            # time.sleep(2)
        # else:
            # time.sleep(0.2)

        #os.system('cls')


class DStar(AStar):
    pass
