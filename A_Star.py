from ast import While
import numpy as np
from copy import deepcopy

def list_sort(list_n):
    list_n.sort(key = lambda man_val: man_val[2])
    return list_n

def a_init_aStar(data):
    """Initializes the a_star grid

    Args:
        grid (numpy array): grid with barriers

    Returns:
        numpy array: grid with room for x, y, and manhattan distance
                     of previous tile
    """
    width, height = data[0].split(" ")
    width = int(width)
    height = int(height)

    # 3 because x prev, y prev, manhattan distance
    temp_grid = np.zeros((height, width, 3))
    perm_grid = np.zeros((height, width))

    # separating out the input obstructions
    num_obstructions = int(data[1])
    obstructions = []

    # plus 2 because of the data offset in the input file
    for i in range(2, num_obstructions+2):
        temp = data[i].split()
        obstructions.append(temp)

    obstructions = np.asarray(obstructions)

    wire_location = num_obstructions+2

    num_wires = int(data[wire_location])

    wires_temp = data[wire_location+1:]

    # separating out the wires
    wires = []
    for i in range(num_wires):
        temp = wires_temp[i].split()
        wires.append(temp)
        num_pins = wires[i][0]
        for j in range(int(num_pins)):
            x = int(wires[i][1 + (int)(j) * 2])
            y = int(wires[i][2 + (int)(j) * 2])
            perm_grid[y][x] = (-1)* (i + 2)
    for i in range(num_obstructions):
        perm_grid[int(obstructions[i][1])][int(obstructions[i][0])] = -1

    return perm_grid, temp_grid, wires, num_wires

def a_manhattan_distance(loc1, loc2):
    """[summary]

    Args:
        loc1 (numpy array): source coordinates
        loc2 (numpy array): sink coordinates

    Returns:
        int: manhattan distance
    """
    return (abs(int(loc1[0]) - int(loc2[0])) + abs(int(loc1[1]) - int(loc2[1])))

def a_adjacent(t_grid, x, y, value, sink, source):
    """fills in values of adjacent tiles with value

    Args:
        t_grid ([type]): old grid with the adjacent values filled in
        x (int): the x location of the current grid element
        y (int): the y location of the current grid element
        value (int): value to fill in each adjacent grid element with

    Returns:
        t_grid (numpy array): new grid with the adjacent values filled in
        new_locations (numpy array): x any y coordinates of the newly filled in tiles
    """

    new_locations = []
    # if the point is in bounds, if it's unoccupied, and if it has the same or less manhattan distance
    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1][2] == 0):
        value = a_manhattan_distance(source, [y, x + 1])
        t_grid[y][x + 1] = [x, y, value]
        temp = [x + 1, y, value]
        new_locations += temp
    if( (x - 1) >= 0 and t_grid[y][x - 1][2] == 0):
        value = a_manhattan_distance(source, [y, x - 1])
        t_grid[y][x - 1] = [x, y, value]
        temp = [x - 1, y, value]
        new_locations += temp
    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x][2] == 0):
        value = a_manhattan_distance(source, [y + 1, x])
        t_grid[y + 1][x] = [x, y, value]
        temp = [x, y + 1, value]
        new_locations += temp
    if( (y - 1) >= 0 and t_grid[y - 1][x][2] == 0):
        value = a_manhattan_distance(source, [y - 1, x])
        t_grid[y - 1][x] = [x, y, value]
        temp = [x, y - 1, value]
        new_locations += temp
    return np.asarray(t_grid), new_locations


# shouldn't need to modify this one too much. might need to check for the lowest value one,
# but I don't think so.
def a_found_sink(t_grid, found_connection, x, y, j, con_loc):
    """looks for sinks in adjacent squares at the current grid tile
       Used in the forward pass

    Args:
        t_grid (numpy array): grid with the distances from the source
        found_connection (numpy array): value at the found connection
        x (int): x coordinate of current grid tile
        y (int): y coordinate of current grid tile
        j (int): the number of the sink we're looking at
        con_loc (np array): the tile of the found connection

    Returns:
        connection_location [numpy array]: location of the tile closest to the sink (if connected to net)
        found_connection [int]: value of the tile closest to the sink (if connected to net)
        boolean: if we have found all of the sinks
    """
    num_pins = len(found_connection)

    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1][2] > 0):
        found_connection[j] = t_grid[y][x + 1][2]
        con_loc[j] = [x + 1, y]
    if( (x - 1) >= 0 and t_grid[y][x - 1][2] > 0):
        found_connection[j] = t_grid[y][x - 1][2]
        con_loc[j] = [x - 1, y]
    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x][2] > 0):
        found_connection[j] = t_grid[y + 1][x][2]
        con_loc[j] = [x, y + 1]
    if( (y - 1) >= 0 and t_grid[y - 1][x][2] > 0):
        found_connection[j] = t_grid[y - 1][x][2]
        con_loc[j] = [x, y - 1]

    return con_loc, found_connection, (np.count_nonzero(found_connection) >= num_pins)

def a_backtrace(t_grid, perm_grid, sink_locations, wire_num):
    print("entering backtrace...")
    wire_value = -1 * (wire_num + 2)

    # avoid connections only between sinks - need to connect to source

    for sink in range(int(len(sink_locations)/2)):
        x = int(sink_locations[int(sink) * 2])
        y = int(sink_locations[1 + int(sink) * 2])
        perm_grid[y][x] = 999999

    # for each sink
    for sink in range(len(sink_locations)//2):
        
        x = int(sink_locations[int(sink) * 2])
        y = int(sink_locations[1 + int(sink) * 2])
        print("y: " + str(y) + ", x: " + str(x))
        while True:
            # if we've reached a wire that is not the original source
            if(perm_grid[y][x] == wire_value): 
                break
            else:
                perm_grid[y][x] = wire_value
                x, y = t_grid[y][x][:2]
                x = int(x)
                y = int(y)
    
    # change the sink values back to their original values
    for sink in range(len(sink_locations)//2):
        x = int(sink_locations[int(sink) * 2])
        y = int(sink_locations[1 + int(sink) * 2])
        perm_grid[y][x] = wire_value
    
    return perm_grid



def a_solve(perm_grid, temp_grid, wires, num_wires):
    for wire in range(num_wires):
        sinks_found = False

        temp_grid[temp_grid > 0] = 0

        updated_locations = wires[wire][3:5]
        source = wires[wire][1:3]

        # should be able to substitute below values with temp_grid

        # used to keep track of if we have a connection between the new "path"
        # of values between the souce and the sink. Stores the numerical value
        # of the first grid location that arrives adjacent to the pin in question
        connection = np.zeros((int((wires[wire][0])) - 1))

        # Keeps track of the location (x, y) of each saved connection
        connection_location = np.zeros((int((wires[wire][0])) - 1, 2))
        # first pin is a source
        print("number of sinks: " + str(int(wires[0][0]) - 1))
        for sink in range (int(wires[0][0]) - 1):
            
            while len(updated_locations):
                print("made it here")
                # initial manhattan distance
                x = int(updated_locations[0])
                y = int(updated_locations[1])
                man_dist = a_manhattan_distance(source, [x, y])
                temp_grid, temp_locations = a_adjacent(temp_grid, x, y, man_dist, sink, source)
                print(temp_grid)
                # bring the lowest manhattan distances to the front
                updated_locations = list_sort(updated_locations + temp_locations)
                connection_location, connection, sinks_found = a_found_sink(temp_grid, connection, x, y, int(sink), connection_location)
                
                if(sinks_found):
                    break
                # this might not be necessary - I think the updated locations will catch it
                if(man_dist > perm_grid[0] + perm_grid[1]):
                    print("Wire " + int(sink) + " not found.")
                    break
            sinks = wires[wire][3:]
            perm_grid = a_backtrace(temp_grid, perm_grid, sinks, int(wire))
    return perm_grid