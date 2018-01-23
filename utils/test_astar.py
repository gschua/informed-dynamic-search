from dstar import DStar
import array

# graph = array.array('f',
#     [0.10, 0.30, 0.45, 0.00, 0.16, 0.26, 0.70, -1.00, 0.63, -1.00, 0.54, -1.00, -1.00, 0.63, 0.72, 0.09, 0.23, 0.8, 0.81, 0.42])
# graph = [0.10, 0.30, 0.45, 0.00, 0.16, 0.26, 0.70, -1.00, 0.63, -1.00, 0.54, -1.00, -1.00, 0.63, 0.72, 0.09, 0.23, 0.8, 0.81, 0.42]
graph = [0.10, 0.30, 0.50, 0.00, 0.16, 0.26, 0.70, -1.00, 0.63, -1.00, 0.54, -1.00, -1.00, 0.63, 0.72, 0.09, 0.23, 0.8, 0.81, 0.42]

ds = DStar(graph, 5, 4, (0, 0), (4, 3), debug=False)
a, b = ds.search()
print(a)
print(b)


ds = DStar([], 548, 421, (142, 328), (125, 375))
print('Start: {}'.format(ds.tuple_to_array(142, 328)))
print('Goal: {}'.format(ds.tuple_to_array(125, 375)))
print()
print(ds.array_to_tuple(180408))
print(ds.array_to_tuple(179860))

