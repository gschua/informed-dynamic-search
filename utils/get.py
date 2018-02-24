import array
import sys
from init import *

def get(day, hour, index):
    with open(get_file(day, hour), 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                print(line.strip())
                return

def tuple_to_index(xcoor, ycoor):
    assert xcoor < xsize, 'Invalid x-coordinate: {}'.format(xcoor)
    assert ycoor < ysize, 'Invalid y-coordinate: {}'.format(ycoor)
    if zero_index:
        assert xcoor >= 0, 'Invalid x-coordinate: {}'.format(xcoor)
        assert ycoor >= 0, 'Invalid y-coordinate: {}'.format(ycoor)
        return ycoor * xsize + xcoor
    else:
        assert xcoor >= 1, 'Invalid x-coordinate: {}'.format(xcoor)
        assert ycoor >= 1, 'Invalid y-coordinate: {}'.format(ycoor)
        return (ycoor-1) * xsize + (xcoor-1)

if __name__ == '__main__':
    day = int(sys.argv[2])
    hour = int(sys.argv[3])
    if len(sys.argv) == 5:
        index = int(sys.argv[4])
    else:
        index = tuple_to_index(int(sys.argv[4]), int(sys.argv[5]))
    get(day, hour, index)


