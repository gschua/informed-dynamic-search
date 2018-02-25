'''To-do:
 - Tkinter GUI for map
 - Proper cost limit checking
 - Float support for costs/graphs
 - test what if zero_index = True
 - goes nuts with 0 cost edges
'''

import copy
import heapq
import itertools
import time
import os
import datetime
import sys

# lol debugging
import pdb
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


def get_time(last_time):
    def clean(t):
        return t.strftime('%H:%M:%S')
    now = datetime.datetime.now().replace(microsecond=0)
    print('Took {}'.format(now-last_time))
    return now


class PriorityQueue:

    '''Implementation of extra functions required to extend heapq to a better
    prioirty queue. Taken from https://docs.python.org/3.6/library/heapq.html#priority-queue-implementation-notes'''

    # placeholder for a removed task
    REMOVED = '<removed-item>'

    def __init__(self):
        # heap is a list of tuples of (priority, node index)
        self.heap = []
        self.indexer = {}
        self.counter = itertools.count()

    def len(self):
        return len(self.indexer)

    def has_node(self, node):
        return node in self.indexer

    def push(self, priority, item):
        if item in self.indexer:
            entry = self.indexer.pop(item)
            entry[-1] = PriorityQueue.REMOVED
        count = next(self.counter)
        entry = [priority, count, item]
        self.indexer[item] = entry
        heapq.heappush(self.heap, entry)

    def pop(self):
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item != PriorityQueue.REMOVED:
                del self.indexer[item]
                return item
        return None


class Search:

    '''Assumptions:
     - each node in the graph has edges only with its 4-adjacent neighbors
     - the cost of going from a node A to a node B is the the same regardless
       of A. B is the only factor in cost.
    presumably it is trivial to extend this function to general graphs but maybe some other time
    '''

    def __init__(self, xsize, ysize, start, goal, zero_index, heuristic_type, limit_type=None, graph_output=None):
        '''Args:

            xsize           int. Horizontal size of graph.

            ysize           int. Vertical size of graph.

            start           (int, int). The x, y coordinates of the starting
                            point.

            goal            (int, int). The x, y coordinates of the goal point.

            zero_index      bool. States whether the graph's origin is at
                            (0, 0) (True) or (1, 1) (False)

            heuristic_type  str. Dictates that heuristic algorithm to be used.
                            Options:

                "a"         A*
                "d"         Dijkstra

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

        if heuristic_type == 'a':
            self.heuristic = self.astar_heuristic
        elif heuristic_type == 'd':
            self.heuristic = self.dijkstra_heuristic
        else:
            raise ValueError('heuristic_type must be either "a" or "d"')

        self.limit_type = limit_type
        assert limit_type == 'c' or limit_type == 'n' or not limit_type

        # check with arbitrary values if graph_output is valid now
        # instead of later after the seacrh is complete hehe
        self.graph_output = graph_output
        if graph_output:
            try:
                test = graph_output.format(g=0, t=0)
            except:
                raise ValueError('Expecting a string for formatting with parameters d and t')

        self.node_count = self.xsize * self.ysize
        self.start_index = self.tuple_to_index(*start)
        self.goal_index = self.tuple_to_index(*goal)
        self.map_limit = -1
        self.cost_threshold = -1

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
        '''

        self.nodes = [
            {
                'total_cost': -1,
                'visited': False,
                'parent': -1,
                'distance': -1,
            } for i in range(self.node_count)
        ]

        # init some parameters for the starting node
        self.nodes[self.start_index].update({
            'total_cost': self.get_cost(self.start_index, self.start_index),
            # i have no idea why but limit doesnt work for the first map unless we set start distance to 2
            'distance': 0,
        })

        self.frontier = PriorityQueue()
        self.frontier.push(0, self.start_index)

    def set_graph(self, graph):
        if len(graph) != self.node_count:
            raise ValueError(
                'Expected graph to be of length {0}, got graph of length {1}'
                .format(self.node_count, len(graph)))
        self.graph = graph
        #self.heuristic_constant = self.get_heuristic_constant()
        if self.cost_threshold < 0:
            self.cost_threshold = max(self.graph) + 1

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
            return self.graph[next]
        return graph[next]

    def astar_heuristic(self, node):
        # to-do: try different formula
        xcoor, ycoor = self.index_to_tuple(node)
        return ((abs(self.goal_tuple[0] - xcoor) + abs(self.goal_tuple[1] - ycoor))
            #* self.heuristic_constant + self.nodes[node]['total_cost'])
            + self.nodes[node]['total_cost'])

    def dijkstra_heuristic(self, node):
        # decide to go with dijkstra because we dont want greedy search for
        # limited step maps
        return self.nodes[node]['total_cost']

    def tuple_to_index(self, xcoor, ycoor):
        #assert xcoor < self.xsize, 'Invalid x-coordinate: {}'.format(xcoor)
        #assert ycoor < self.ysize, 'Invalid y-coordinate: {}'.format(ycoor)
        if self.zero_index:
            #assert xcoor >= 0, 'Invalid x-coordinate: {}'.format(xcoor)
            #assert ycoor >= 0, 'Invalid y-coordinate: {}'.format(ycoor)
            return ycoor * self.xsize + xcoor
        else:
            #assert xcoor >= 1, 'Invalid x-coordinate: {}'.format(xcoor)
            #assert ycoor >= 1, 'Invalid y-coordinate: {}'.format(ycoor)
            return (ycoor-1) * self.xsize + (xcoor-1)

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

    def set_node(self, parent, child, new_cost):

        # assign the new parent's values
        self.nodes[child]['total_cost'] = new_cost
        self.nodes[child]['parent'] = parent
        self.nodes[child]['distance'] = self.nodes[parent]['distance'] + 1

    def search(self):

        count = 0
        found = False

        while True:

            current = self.frontier.pop()
            # no more nodes to investigate
            if current == None:
                break

            count += 1
            self.nodes[current]['visited'] = True
            #self.mini_illustrate(current)

            # finish if we find reach the goal
            if current == self.goal_index:
                found = True
                break
            # reached max visitable number of nodes
            if self.limit_type == 'n' and self.nodes[current]['distance'] >= self.map_limit:
                continue

            for next in self.get_neighbors(current):

                    # node is closed
                if (self.nodes[next]['visited'] or
                    # neighbor is an obstacle, not node
                    self.graph[next] > self.cost_threshold):
                    continue

                new_cost = self.nodes[current]['total_cost'] + self.get_cost(current, next)
                # if self.limit_type == 'c' and self.map_limit <= new_cost:
                    # continue
                if self.nodes[next]['total_cost'] < 0 or self.nodes[next]['total_cost'] > new_cost:
                    # set new node values
                    self.set_node(current, next, new_cost)
                    # update priority queue
                    self.frontier.push(self.heuristic(next), next)

        if self.graph_output:
            self.illustrate()

        print('Searched {} nodes'.format(count))
        return found

    def final_path(self):

        path = []
        traversed = []
        current = self.goal_index
        child = None

        while current >= 0:
            # check for cycles
            if current in traversed:
                pdb.set_trace()
                raise Exception('Cyclic path found!')

            path.append((current, self.nodes[current]))
            traversed.append(current)
            child = current
            current = self.nodes[current]['parent']

        # reverse list so it goes from start -> goal instead of goal -> start
        path.reverse()
        return path

    def mini_illustrate(self, current):

        os.system('cls')
        row = 0
        max = 61
        for i in range(self.node_count):
            xcoor, ycoor = self.index_to_tuple(i)
            if xcoor < self.start_tuple[0] - 30 or xcoor > self.start_tuple[0] + 30 or ycoor < self.start_tuple[1] - 10 or ycoor > self.start_tuple[1] + 30:
                continue
            if i == self.start_index:
                sys.stdout.write('o ')
            elif i == self.goal_index:
                sys.stdout.write('x ')
            elif i == current:
                sys.stdout.write('# ')
            elif self.nodes[i]['visited']:
                sys.stdout.write('+ ')
            elif self.graph[i] > self.cost_threshold:
                sys.stdout.write('( ')
            else:
                sys.stdout.write('. ')
            row += 1
            if row >= max:
                row = 0
                sys.stdout.write('\n')
        x = input()

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

class DynamicSearch(Search):

    '''
    Although originally called DStar, this implementation is quite different
    from the traditional D* problem in that

    1. it deals with known graph changes rather than incomplete information and
    2. it expects some form of distance or cost limitation to be reached before
    performing a graph switch

    This implementation maintains the original path cost if it is lower
    than the new cost.

    Features:
     - Basic Map Illustration (Plain Text)
     - Trigger map change based on either cost or number of visited nodes.
    '''

    def __init__(self, graph_set, path_output, cost_threshold_list, map_limits, *args, **kwargs):

        super(DStar, self).__init__(*args, **kwargs)

        self.graph_set = graph_set
        self.map_limits = map_limits

        # check with arbitrary values if path_output is valid now
        self.path_output = path_output
        try:
            test = path_output.format(t=0)
        except:
            raise Exception('Expecting a string for formatting with parameter t')
        if cost_threshold_list:
            self.cost_threshold_list = cost_threshold_list
        else:
            self.cost_threshold_list = [self.max_cost + 1]

        self.max_cost = max(max(g) for g in graph_set)
        self.last_time = datetime.datetime.now().replace(microsecond=0)

    def reset(self):

        super(DStar, self).reset()
        self.dnodes[self.graph_number][self.start_index] = {
            'total_cost': self.get_cost(self.start_index, self.start_index),
            'distance': 0,
            'parent': -1,
            'parent_map': -1,
            'ancestors': [],
        }

    def set_node(self, parent, child, new_cost):

        super(DStar, self).set_node(parent, child, new_cost)
        #parent_info = self.dnodes[self.nodes[parent]['map']][parent]
        parent_info = self.dnodes[self.graph_number][parent]
        self.dnodes[self.graph_number][child] = {
            'total_cost': new_cost,
            'distance': parent_info['distance'] + 1,
            'parent': self.nodes[child]['parent'],
            #'parent_map': self.nodes[parent]['map'],
            'parent_map': self.graph_number,
            'ancestors': parent_info['ancestors'] + [parent],
        }

    def astar_heuristic(self, node, graph_number=None):
        if graph_number == None:
            graph_number = self.graph_number
        xcoor, ycoor = self.index_to_tuple(node)
        return ((abs(self.goal_tuple[0] - xcoor) + abs(self.goal_tuple[1] - ycoor))
            + self.dnodes[graph_number][node]['total_cost'])

    def dijkstra_heuristic(self, node, graph_number=None):
        # decide to go with dijkstra because we dont want greedy search for
        # limited step maps
        if graph_number == None:
            graph_number = self.graph_number
        return self.dnodes[graph_number][node]['total_cost']

    def write_path(self, path, cost):

        output_file = self.path_output.format(t=str(self.cost_threshold).zfill(2))
        print('\nPath found! Writing path to {}'.format(output_file))
        idx_list = []
        with open(output_file, 'w') as f:

            f.write('Cost: {}\n'.format(cost))
            f.write('Threshold: {}/{}\n'.format(self.cost_threshold, self.max_cost))
            f.write('index,xcoor,ycoor,map,cost,total_cost,parent,distance\n')

            for node_info in path:
                xcoor, ycoor = self.index_to_tuple(node_info['index'])
                f.write(','.join((
                    str(node_info['index']),
                    str(xcoor),
                    str(ycoor),
                    str(node_info['map']),
                    str(node_info['cost']),
                    str(node_info['total_cost']),
                    str(node_info['parent']),
                    str(node_info['distance']),
                )))
                f.write('\n')
                idx_list.append(node_info['index'])

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

    def get_wait_penalty(self, node):

        '''When comparing the cost of traversing a node between two maps, we
        need to take into consideration a penalty for waiting for a map change
        instead of taking the extra cost of a less favorable map layout. Thus,
        we add a waiting penalty to all total_costs as later on they will be
        used for waiting vs moving cost comparison.'''

        # Map limit in prev map - number of nodes traversed in previous map = number of steps spent waiting at a node
        # number of steps spent waiting at a node * cost of waiting once in the previous map = total cost of waiting N steps at a node
        # total cost of waiting N steps at a node + cost of being in the new node = total wait penalty
        prev_map = self.graph_number - 1
        return (self.map_limits[prev_map] - self.nodes[node]['distance']) * self.graph_set[prev_map][node]

    #@profile
    def reevaluate(self):

        # very similar to astar_search(), except that for every pair of visited
        # nodes, we check the new cost of traversing

        count = 0
        reev_heap = PriorityQueue()
        reevaluated = []
        waiting_cost = []

        # init reev_heap, reevaluated, and waiting_cost
        for i in range(len(self.nodes)):
            reevaluated.append(False)
            if self.nodes[i]['visited']:
                reev_heap.push(self.heuristic(i, self.graph_number-1), i)
                waiting_cost.append(self.dnodes[self.graph_number-1][i]['total_cost'] + self.get_wait_penalty(i))
            else:
                waiting_cost.append(-1)

        while True:

            current = reev_heap.pop()
            # reev_heap is empty
            if current == None:
                break
            # node is closed
            if reevaluated[current]:
                continue
            '''
            # parent not yet evaluated
            # shove current back into queue with the parent's prioirity
            ancestor = self.nodes[current]['parent']
            for parent in reversed(self.nodes[current]['parents']):
                if not reevaluated[parent]:
                    reev_heap.push(self.heuristic(parent), current)
                    continue
            '''

            count += 1
            reevaluated[current] = True
            new_cost = waiting_cost[current]

            # old_cost can either be
            if current in self.dnodes[self.graph_number]:
                # total_cost of a new more optimal parent than the one found in AStar.search()
                old_cost = self.dnodes[self.graph_number][current]['total_cost']
                parent_map = self.graph_number
            else:
                #self.dnodes[self.graph_number][current] = {}
                # total_cost calculated in the most recent iteration of AStar.search()
                old_cost = self.dnodes[self.graph_number-1][current]['total_cost']
                self.dnodes[self.graph_number][current] = copy.deepcopy(self.dnodes[self.graph_number-1][current])
                # we know change it to what it would be with an added wait penalty
                #self.dnodes[self.graph_number][current]['total_cost'] = new_cost
                parent_map = self.dnodes[self.graph_number-1][current]['parent_map']

            if parent_map < self.graph_number or new_cost < old_cost:
                # 3 scenarios:
                # 1. "current" is the start node
                # 2. None of "current"'s ancestors have a lower cost in the new map
                # 3. It is more efficient for "current" to wait than to come from its parent
                # 
                # now total_cost is the cost if we waited until right before
                # the map change
                # altho technically this node is still in the previous map
                # rather than current map, we set this to current map for
                # set_node() later
                self.dnodes[self.graph_number][current]['total_cost'] = new_cost
                self.nodes[current]['distance'] = 0

            # have hit max step limit
            if self.limit_type == 'n' and self.map_limit <= self.nodes[current]['distance']: continue

            for next in self.get_neighbors(current):

                    # node is closed
                if (reevaluated[next] or
                    # neighbor is an obstacle, not node
                    self.graph[next] > self.cost_threshold or
                    # cycle detected
                    next in self.dnodes[self.graph_number][current]['ancestors']):
                    continue

                new_next_cost = self.nodes[current]['total_cost'] + self.get_cost(current, next)
                # if self.limit_type == 'c' and self.map_limit <= new_next_cost: continue
                if not next in self.dnodes[self.graph_number] or self.dnodes[self.graph_number][next]['total_cost'] > new_next_cost:
                    self.set_node(current, next, new_next_cost)
                    if self.nodes[next]['visited']:
                        reev_heap.push(self.heuristic(next), next)
                    else:
                        self.frontier.push(self.heuristic(next), next)

        print('Re-evaluated {} nodes'.format(count), end=', ', flush=True)

    def final_path(self):

        path = []
        traversed = []
        node = self.goal_index
        map = self.graph_number

        while node >= 0:

            # check for cycles
            if node in traversed:
                pdb.set_trace()
                raise Exception('Cyclic path found!')

            path.append({
                'index': node,
                'map': map,
                'cost': self.graph_set[map][node],
                'total_cost': self.dnodes[map][node]['total_cost'],
                'parent': self.dnodes[map][node]['parent'],
                'parent_map': self.dnodes[map][node]['parent_map'],
                'distance': self.dnodes[map][node]['distance'],
            })
            node = path[-1]['parent']
            map = path[-1]['parent_map']

        # reverse list so it goes from start -> goal instead of goal -> start
        path.reverse()
        return path, path[-1]['total_cost']

    def dynamic_search(self):

        for cost_threshold in sorted(self.cost_threshold_list):

            print('\nThreshold set to {}'.format(cost_threshold))
            self.set_cost_threshold(cost_threshold)
            self.dnodes = {}
            # dnodes[map_number][node_index] = {total_cost, distance, parent, parent's map, ancestors}

            for graph_number, this_graph in enumerate(self.graph_set):

                print('Map change to {}'.format(graph_number), end=', ', flush=True)
                self.graph_number = graph_number
                self.set_graph(this_graph)
                self.set_map_limit(self.map_limits[graph_number])
                self.dnodes[graph_number] = {}

                if graph_number > 0:
                    self.reevaluate()
                else:
                    self.reset()

                print('Performing A* Search', end=', ', flush=True)
                goal_reached = self.search()

                if goal_reached:
                    path, cost = self.final_path()
                    self.write_path(path, cost)
                    return

            self.last_time = get_time(self.last_time)
            print('Path not found, retrying with higher threshold')

        print('\nFailed to find path to goal')
