'''To-do:
 - Tkinter GUI for map
 - Proper cost limit checking
 - Float support for costs/graphs
'''

import copy
import heapq
import time
import os
import datetime
import sys

# lol debugging
import pdb
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

def get_time(last_time, new_line=True):

    def clean(t):
        return t.strftime('%H:%M:%S')

    now = datetime.datetime.now().replace(microsecond=0)
    if new_line:
        print('Took {}'.format(now-last_time))
    else:
        print('Took {}'.format(now-last_time), end=', ', flush=True)
    return now


class AStar:

    '''Subset of Graph, such that it is a graph whose edges are only with its
    4-adjacent neighbors'''

    def __init__(self, xsize, ysize, start, goal, zero_index, limit_type=None, graph_output=None):
        '''Args:

            xsize           int. Horizontal size of graph.

            ysize           int. Vertical size of graph.

            start           (int, int). The x, y coordinates of the starting
                            point.

            goal            (int, int). The x, y coordinates of the goal point.

            zero_index      bool. States whether the graph's origin is at
                            (0, 0) (True) or (1, 1) (False)

            limit_type      str. Dictates that type of limit the traversal is
                            under if any. Options:
                "n"         node. Limits the number of nodes a path encompasses.
                "c"         cost. Limits the total cost of a path.

            graph_output    str. A formattable string with parameters d (for
                            graph number) and t (for threhold). States the
                            file path to place the graph illustrations. If
                            None, no illustration will be generated.
        '''

        self.xsize = xsize
        self.ysize = ysize
        # Internally, a MxN 2D grid is represented as a 1D array/list of length
        # MxN. All calculations are done using the index format of the
        # coordinates rather than the tuple version.
        self.start_tuple = start
        self.goal_tuple = goal
        self.zero_index = zero_index
        self.limit_type = limit_type
        assert limit_type == 'c' or limit_type == 'n' or not limit_type

        # check with arbitrary values if graph_output is valid now
        # instead of later after the seacrh is complete hehe
        self.graph_output = graph_output
        if self.graph_output:
            try:
                test = graph_output.format(g=0, t=0)
            except:
                raise ValueError('Expecting a string for formatting with parameters d and t')

        self.node_count = self.xsize * self.ysize
        self.start_index = self.tuple_to_index(*start)
        self.goal_index = self.tuple_to_index(*goal)
        self.map_limit = -1
        self.cost_threshold = None

    def reset(self):

        '''NOTE: MUST be called only after initially calling set_graph()
        
        For every node, we must keep track of certain parameters.
        
        total_cost      int. Total cost of the most optimal found path to this
                        node, including this node's cost.

        visited         bool. Whether or not this node has been investigated.

        parent          int. Index of the parent node.

        distance        int. Number of nodes from the start. Used for checking
                        path length limits. Starts from 1, e.g. the start node
                        has a distance of 1, the next node has a distance of 2,
                        etc.

        D* Exclusive Parameters:

        map         int. The currently most optimal map to visit this node.
        edge_cost   int. The cost of reaching this node from "parent" given 
                    graph "map".
        parent_map  int. The "map" value when the edge between "parent" and the
                    current node was crossed. Since it is possible for "parent"
                    to be assigned to the current node during map N and later
                    change its values during map N+M, we need to store the map
                    value at the point when the edge between "parent" and the
                    current node was crossed.
        parents     list of int. A list of ancestors of the current node.
                    Required to avoid getting into cycles.

        real_distance
        '''

        self.nodes = [
            {
                'total_cost': -1,
                'visited': False,
                'parent': -1,
                'distance': -1,

                'map': -1,
                'real_distance': -1,
                'parent_map': -1,
                'edge_cost': -1,
                'parents': [],
            } for i in range(self.node_count)
        ]

        # init some parameters for the starting node
        self.nodes[self.start_index]['total_cost'] = self.get_cost(
            self.start_index, self.start_index)
        self.nodes[self.start_index]['distance'] = 1
        self.nodes[self.start_index]['map'] = self.graph_number
        self.nodes[self.start_index]['real_distance'] = 1
        self.nodes[self.start_index]['edge_cost'] = copy.deepcopy(
            self.nodes[self.start_index]['total_cost'])

        self.frontier = [(0, self.start_index)]   # (priority, node)

    def set_graph(self, graph, graph_number=1):
        if len(graph) != self.node_count:
            raise ValueError(
                'Expected graph to be of length {0}, got graph of length {1}'
                .format(self.node_count, len(graph))
            )
        self.graph = graph
        self.graph_number = graph_number
        self.heuristic_constant = self.get_heuristic_constant()

    def set_cost_threshold(self, cost_threshold):
        # since there may be some nodes whose costs are so high as to be
        # undesirable to visit, it may be more 
        self.cost_threshold = cost_threshold

    def set_map_limit(self, map_limit):
        # map limits are used to trigger graph changes. Range is [0, map_limit)
        self.map_limit = map_limit

    def get_heuristic_constant(self):
        # to-do: try different formula
        sum = 0
        for g in self.graph:
            sum += g
        return int(sum/self.node_count)

    def get_cost(self, current, next, graph=None):
        if not graph:
            graph = self.graph
        # to-do: try different formula
        return graph[next]

    def heuristic(self, node):
        xcoor, ycoor = self.index_to_tuple(node)
        # to-do: try different formula
        return (
            (abs(self.goal_tuple[0] - xcoor) + abs(self.goal_tuple[1] - ycoor))
            * self.heuristic_constant + self.nodes[node]['total_cost']
        )

    def tuple_to_index(self, xcoor, ycoor):
        #assert xcoor < self.xsize, 'Invalid x-coordinate: {}'.format(xcoor)
        #assert ycoor < self.ysize, 'Invalid y-coordinate: {}'.format(ycoor)
        if self.zero_index:
            #assert xcoor >= 0, 'Invalid x-coordinate: {}'.format(xcoor)
            #assert ycoor >= 0, 'Invalid y-coordinate: {}'.format(ycoor)
            return ycoor * self.xsize + xcoor
            #return xcoor * self.ysize + ycoor
        else:
            #assert xcoor >= 1, 'Invalid x-coordinate: {}'.format(xcoor)
            #assert ycoor >= 1, 'Invalid y-coordinate: {}'.format(ycoor)
            return (ycoor-1) * self.xsize + (xcoor-1)
            #return (xcoor-1) * self.ysize + (ycoor-1)

    def index_to_tuple(self, idx):
        #assert idx < self.node_count and idx >= 0, 'Invalid index: {}'.format(idx)
        if self.zero_index:
            return idx % self.xsize, int(idx/self.xsize)
        else:
            return idx % self.xsize + 1, int(idx/self.xsize) + 1

    def get_neighbors(self, center):

        def north_exists():
            try:
                return north > -1
            except:
                return False
        def south_exists():
            try:
                return south < self.node_count
            except:
                return False
        def east_exists():
            try:
                return int(east/self.ysize) == int(center/self.ysize)
            except:
                return False
        def west_exists():
            try:
                return int(west/self.ysize) == int(center/self.ysize) and west > -1
            except:
                return False

        # center should be an int of range [0, self.node_count)
        # but adding a check wastes time

        neighbors = []

        north = center - self.xsize
        south = center + self.xsize
        east = center + 1
        west = center - 1

        if north_exists():
            neighbors.append(north)
        if south_exists():
            neighbors.append(south)
        if east_exists():
            neighbors.append(east)
        if west_exists():
            neighbors.append(west)

        return neighbors

    #@profile
    def pq_push(self, priority_queue, node):

        priority = self.heuristic(node)
        for n in priority_queue:
            if n[1] == node:
                n = (priority, node)
                heapq.heapify(priority_queue)
                return
        heapq.heappush(priority_queue, (priority, node))

    def set_parent(self, parent, child):

        # assign the new parent's values
        self.nodes[child]['parent'] = parent
        self.nodes[child]['parents'] = self.nodes[parent]['parents'] + [parent]
        self.nodes[child]['parent_map'] = copy.deepcopy(self.nodes[parent]['map'])
        self.nodes[child]['distance'] = self.nodes[parent]['distance'] + 1
        self.nodes[child]['edge_cost'] = self.get_cost(parent, child)
        self.nodes[child]['real_distance'] = self.nodes[parent]['real_distance'] + 1

    def astar_search(self):

        count = 0
        while self.frontier:

            priority, current = heapq.heappop(self.frontier)

            count += 1
            self.nodes[current]['visited'] = True

            # finish if we find reach the goal
            if current == self.goal_index:
                break
            # if we have reached the maximum number of nodes we can visit, we
            # will not investigate this node
            if self.limit_type == 'n' and self.nodes[current]['distance'] >= self.map_limit:
                continue

            for next in self.get_neighbors(current):

                # if we already visited the node or it is above cost therehsold
                if self.nodes[next]['visited'] or self.graph[next] > self.cost_threshold:
                    continue

                new_cost = self.nodes[current]['total_cost'] + self.get_cost(current, next)
                # if self.limit_type == 'c' and self.map_limit <= new_cost:
                    # continue
                if self.nodes[next]['total_cost'] < 0 or self.nodes[next]['total_cost'] > new_cost:
                    # set new cost and map number
                    self.nodes[next]['total_cost'] = new_cost
                    self.nodes[next]['map'] = self.graph_number
                    # change next's parent to current
                    self.set_parent(current, next)
                    # update priority queue
                    self.pq_push(self.frontier, next)

        if self.graph_output:
            self.illustrate()

        print('Searched {} nodes'.format(count), end=', ', flush=True)
        return self.final_path(), self.nodes[self.goal_index]['total_cost']

    def final_path(self):

        path = []
        traversed = []
        idx = self.goal_index
        prev = None

        while idx >= 0 and idx != self.start_index:
            # check for cycles
            if idx in traversed:
                raise Exception('Cyclic path found!')

            # determine the map and edge cost of the current node at the time
            # this optimal path crosses it
            if prev:
                self.nodes[idx]['actual_map'] = self.nodes[prev]['parent_map']
                self.nodes[idx]['actual_edge_cost'] = self.get_cost(idx, prev, self.graph_set[self.nodes[idx]['actual_map']-self.graph_num_start])
            else:
                self.nodes[idx]['actual_map'] = self.graph_number
                self.nodes[idx]['actual_edge_cost'] = self.get_cost(self.nodes[idx]['parent'], idx)

            path.append((idx, self.nodes[idx]))
            traversed.append(idx)
            prev = copy.deepcopy(idx)
            idx = self.nodes[idx]['parent']

        # one more time or it won't include starting node
        self.nodes[idx]['actual_map'] = self.nodes[prev]['parent_map']
        self.nodes[idx]['actual_edge_cost'] = self.get_cost(idx, prev, self.graph_set[self.nodes[idx]['actual_map']-self.graph_num_start])
        path.append((idx, self.nodes[idx]))
        # reverse list so it goes from start -> goal instead of goal -> start
        path.reverse()

        return path

    def illustrate(self):

        output_file = self.graph_output.format(
            g=str(self.graph_number).zfill(2),
            t=str(self.cost_threshold).zfill(2),
        )
        with open(output_file, 'w') as f:
            row = 0
            for i in range(self.node_count):
                if i == self.start_index:
                    f.write('o ')
                elif i == self.goal_index:
                    f.write('x ')
                elif self.nodes[i]['visited'] and self.graph[i] > self.cost_threshold:
                    f.write('# ')
                elif self.nodes[i]['visited']:
                    f.write('+ ')
                elif self.graph[i] > self.cost_threshold:
                    f.write('~ ')
                else:
                    f.write('. ')
                row += 1
                if row >= self.xsize:
                    row = 0
                    f.write('\n')

class DStar(AStar):

    '''
    Although called DStar, this implementation is quite different from the
    traditional D* problem in that

    1. it deals with known graph changes rather than incomplete information and
    2. it expects some form of distance or cost limitation to be reached before
    performing a graph switch

    This implementation maintains the original path cost if it is lower
    than the new cost.

    Features:
     - Basic Map Illustration (Plain Text)
     - Trigger map change based on either cost or number of visited nodes.
    '''

    def __init__(self, graph_set, max_cost, path_output, cost_threshold_list, map_limits, graph_num_start, *args, **kwargs):

        super(DStar, self).__init__(*args, **kwargs)

        self.graph_set = graph_set
        self.max_cost = max_cost
        self.map_limits = map_limits
        self.graph_num_start = graph_num_start

        # check with arbitrary values if path_output is valid now
        self.path_output = path_output
        try:
            test = path_output.format(t=0)
        except:
            raise Exception('Expecting a string for formatting with parameter t')

        if cost_threshold_list:
            self.cost_threshold_list = cost_threshold_list
        else:
            self.cost_threshold_list = [max_cost]

        self.last_time = datetime.datetime.now().replace(microsecond=0)

    def write_path(self, path, cost):

        output_file = self.path_output.format(t=str(self.cost_threshold).zfill(2))
        print('\nPath found! Writing path to {}'.format(output_file))
        idx_list = []
        with open(output_file, 'w') as f:

            f.write('Cost: {}\n'.format(cost))
            f.write('Threshold: {}/{}\n'.format(self.cost_threshold, self.max_cost))
            f.write('index,xcoor,ycoor,map,edge_cost,total_cost,parent,distance,a\n')

            for idx, node_info in path:
                xcoor, ycoor = self.index_to_tuple(idx)
                f.write(','.join((
                    str(idx),
                    str(xcoor),
                    str(ycoor),
                    str(node_info['actual_map']),
                    str(node_info['actual_edge_cost']),
                    str(node_info['total_cost']),
                    str(node_info['parent']),
                    str(node_info['real_distance']),
                    str(node_info['distance'])
                )))
                f.write('\n')
                idx_list.append(idx)

        output_file = '{0}_{2}.{1}'.format(*output_file.rsplit('.', 1), 'graph')
        with open(output_file, 'w') as f:
            row = 0
            for i in range(self.node_count):
                if i == self.start_index:
                    f.write('o ')
                elif i == self.goal_index:
                    f.write('x ')
                elif i in idx_list:
                    f.write('+ ')
                else:
                    f.write('. ')
                row += 1
                if row >= self.xsize:
                    row = 0
                    f.write('\n')

    def start_heuristic(self, node):
        '''similar to AStar.heuristic() except it evaluates based on closeness
        to start node'''
        # who would win
        # a fancy robust heuristic algorithm
        # or sort by cost boi
        return self.nodes[node]['total_cost']

    def get_wait_penalty(self, node):

        '''When comparing the cost of traversing a node between two maps, we
        need to take into consideration a penalty for waiting for a map change
        instead of taking the extra cost of a less favorable map layout. Thus,
        we add a waiting penalty to all total_costs as later on they will be
        used for waiting vs moving.'''
        multiplier = 1
        if self.limit_type == 'n':
            return (self.map_limit - self.nodes[node]['distance']) * multiplier * (self.graph_number - self.nodes[node]['map'])
        else:
            # lol idk work on this if you have an example of cost limited maps
            return multiplier

    #@profile
    def reevaluate(self):

        # very similar to astar_search(), except that for every pair of visited
        # nodes, we check the new cost of traversing

        count = 0
        reev_heap = []
        reevaluated = [False for i in range(self.node_count)]

        for i in range(len(self.nodes)):
            if self.nodes[i]['visited']:
                self.pq_push(reev_heap, i)

        while reev_heap:

            priority, current = heapq.heappop(reev_heap)

            if reevaluated[current]:
                continue

            count += 1
            reevaluated[current] = True
            new_cost = (
                self.nodes[current]['total_cost'] -
                self.nodes[current]['edge_cost'] +
                self.graph[current] +
                self.get_wait_penalty(current)
            )

            # if current == 131733 and self.graph_number == 9:
                # pdb.set_trace()

            if self.nodes[current]['total_cost'] > new_cost:
                self.nodes[current]['total_cost'] = new_cost
                self.nodes[current]['edge_cost'] = self.graph[current]
                self.nodes[current]['map'] = self.graph_number

            for next in self.get_neighbors(current):

                if (reevaluated[next] or
                    next in self.nodes[current]['parents'] or
                    self.graph[next] > self.cost_threshold):
                    continue

                # if next in range(139943, 139946) and self.graph_number == 9:
                    # winsound.Beep(frequency, duration)
                    # pdb.set_trace()

                new_next_cost = self.nodes[current]['total_cost'] + self.get_cost(current, next)
                # if self.limit_type == 'c' and self.map_limit <= new_next_cost:
                    # continue

                if self.nodes[next]['total_cost'] < 0 or self.nodes[next]['total_cost'] > new_next_cost:
                    self.nodes[next]['total_cost'] = new_next_cost
                    self.nodes[next]['map'] = self.graph_number
                    self.set_parent(current, next)

                # have to put it outside the previous if statement since it is
                # possible for current node's cost to not lower but next's
                # cost will anyway
                if self.nodes[next]['visited']:
                    self.pq_push(reev_heap, next)
                else:
                    # do NOT add a node if it we already reached the edge and
                    # it is not a new node
                    if self.limit_type == 'n' and self.nodes[current]['distance'] >= self.map_limit:
                        pass
                    else:
                        self.pq_push(self.frontier, next)

            # if current == 131733 and self.graph_number == 9:
                # winsound.Beep(frequency, duration)
                # pdb.set_trace()

        # reset distance for later wait penalty computation
        for node in self.nodes:
            node['distance'] = 0

        print('Re-evaluated {} nodes'.format(count), end=', ', flush=True)

    def dstar_search(self):

        for cost_threshold in sorted(self.cost_threshold_list):

            print('\nThreshold set to {}'.format(cost_threshold))
            self.set_cost_threshold(cost_threshold)

            for graph_num, this_graph in enumerate(self.graph_set, self.graph_num_start):

                print('Map change to {}'.format(graph_num), end=', ', flush=True)
                self.set_graph(this_graph, graph_num)
                self.set_map_limit(self.map_limits[graph_num - self.graph_num_start])

                if graph_num > self.graph_num_start:
                    self.reevaluate()
                    self.last_time = get_time(self.last_time, new_line=False)
                else:
                    self.reset()

                print('Performing A* Search', end=', ', flush=True)
                path, cost = self.astar_search()
                self.last_time = get_time(self.last_time)

                if cost >= 0:
                    self.write_path(path, cost)
                    return

            print('Path not found, retrying with higher threshold')

        print('\nFailed to find path to goal')
