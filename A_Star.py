from operator import is_not
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import colors

def list_sort(list_n):
    list_n.sort(key = lambda man_val: man_val[2])
    return list_n

def plot_grid(grid_n):
    plt.imshow(grid_n, interpolation='none')
    plt.show()
    plt.clf()

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
    x = (abs(int(loc1[0]) - int(loc2[0])) + abs(int(loc1[1]) - int(loc2[1])))
    #print("man_dist: " + str(x))
    return x

def a_adjacent(t_grid, x, y, value, sink, source, loc_dict, perm_grid, wire, all_sinks, connection, connection_location):
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

    for i in range(int(len(all_sinks) / 2)):
        x_t = int(all_sinks[int(i) * 2])
        y_t = int(all_sinks[1 + int(i) * 2])
        perm_grid[y_t][x_t] = 0

    wire_found = False
    wire_value = -1 * (2 + wire)
    # if we've arrived at a wire that has already been placed
    if ( (x + 1) < t_grid.shape[1] and perm_grid[y][x + 1] == wire_value and not is_sink(x+1, y, all_sinks) ):
        wire_found = True
        value = a_manhattan_distance([x + 1, y], source) + a_manhattan_distance([x + 1, y], sink)
        t_grid[y][x + 1] = [x, y, value]
        connection[0] = value
        connection_location[0] = [x, y]
        if (value in loc_dict):
            loc_dict[value].append(x+1)
            loc_dict[value].append(y)
        else:
            loc_dict[value] = [x+1, y]
    # print(is_sink(x-1, y, all_sinks))
    # print(x)
    # print(y)
    # print(wire_value)
    # print(perm_grid[y][x-1])
    # print(perm_grid)
    # new_grid = []
    # for i in range(len(perm_grid)):
    #     for j in range(len(perm_grid[0])):
    #         new_grid.append(t_grid[i][j][2])
    # new_grid = np.asarray(new_grid)
    # new_grid = np.reshape(new_grid, (len(perm_grid), len(perm_grid[0])))
    # print(new_grid)
    # print(perm_grid[y][x - 1] == wire_value)
    if( (x - 1) >= 0 and perm_grid[y][x - 1] == wire_value and not is_sink(x-1, y, all_sinks)):
        print("i'm in here")
        wire_found = True
        value = a_manhattan_distance([x - 1, y], source) + a_manhattan_distance([x - 1, y], sink)
        t_grid[y][x - 1] = [x, y, value]
        # new_grid = []
        # for i in range(len(perm_grid)):
        #     for j in range(len(perm_grid[0])):
        #         new_grid.append(t_grid[i][j][2])
        # new_grid = np.asarray(new_grid)
        # new_grid = np.reshape(new_grid, (len(perm_grid), len(perm_grid[0])))
        # print(new_grid)
        connection[0] = value
        connection_location[0] = [x, y]
        if (value in loc_dict):
            loc_dict[value].append(x-1)
            loc_dict[value].append(y)
        else:
            loc_dict[value] = [x-1, y]

    if( (y + 1) < t_grid.shape[0] and perm_grid[y + 1][x] == wire_value and not is_sink(x, y + 1, all_sinks)):
        print("idk how but i'm in here")
        wire_found = True
        value = a_manhattan_distance([x, y + 1], source) + a_manhattan_distance([x, y + 1], sink)
        connection[0] = value
        connection_location[0] = [x, y]
        t_grid[y + 1][x] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x)
            loc_dict[value].append(y+1)
        else:
            loc_dict[value] = [x, y+1]

    if( (y - 1) >= 0 and perm_grid[y - 1][x] == wire_value and not is_sink(x, y - 1, all_sinks)):
        print("i'm also in here")
        wire_found = True
        value = a_manhattan_distance([x, y - 1], source) + a_manhattan_distance([x, y - 1], sink)
        connection[0] = value
        connection_location[0] = [x, y]
        t_grid[y - 1][x] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x)
            loc_dict[value].append(y-1)            
        else:
            loc_dict[value] = [x, y-1]

    if (wire_found):
        for i in range(int(len(all_sinks) / 2)):
            x_t = int(all_sinks[int(i) * 2])
            y_t = int(all_sinks[1 + int(i) * 2])
            perm_grid[y_t][x_t] = wire_value
        return np.asarray(t_grid), loc_dict, wire_found, connection, connection_location
    # if the point is in bounds, if it's unoccupied, and if it has the same or less manhattan distance
    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1][2] == 0 and perm_grid[y][x + 1] > -1):
        value = a_manhattan_distance([x + 1, y], source) + a_manhattan_distance([x + 1, y], sink)
        t_grid[y][x + 1] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x+1)
            loc_dict[value].append(y)
        else:
            loc_dict[value] = [x+1, y]

    if( (x - 1) >= 0 and t_grid[y][x - 1][2] == 0 and perm_grid[y][x - 1] > -1):
        value = a_manhattan_distance([x - 1, y], source) + a_manhattan_distance([x - 1, y], sink)
        t_grid[y][x - 1] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x-1)
            loc_dict[value].append(y)
        else:
            loc_dict[value] = [x-1, y]

    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x][2] == 0 and perm_grid[y + 1][x] > -1):
        value = a_manhattan_distance([x, y + 1], source) + a_manhattan_distance([x, y + 1], sink)
        t_grid[y + 1][x] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x)
            loc_dict[value].append(y+1)
        else:
            loc_dict[value] = [x, y+1]

    if( (y - 1) >= 0 and t_grid[y - 1][x][2] == 0 and perm_grid[y - 1][x] > -1):
        value = a_manhattan_distance([x, y - 1], source) + a_manhattan_distance([x, y - 1], sink)
        t_grid[y - 1][x] = [x, y, value]
        if (value in loc_dict):
            loc_dict[value].append(x)
            loc_dict[value].append(y-1)            
        else:
            loc_dict[value] = [x, y-1]
    
    for i in range(int(len(all_sinks) / 2)):
            x_t = int(all_sinks[int(i) * 2])
            y_t = int(all_sinks[1 + int(i) * 2])
            perm_grid[y_t][x_t] = wire_value
    print("wire found: " + str(wire_found))
    return np.asarray(t_grid), loc_dict, wire_found, connection, connection_location

def is_sink(x, y, sinks):
    for i in range(int(len(sinks) / 2)):
        sink_loc = [sinks[i * 2], sinks[i * 2 + 1]]
        if ([x, y] == sink_loc):
            return True
    return False
# shouldn't need to modify this one too much. might need to check for the lowest value one,
# but I don't think so.
def a_found_sink(t_grid, found_connection, source, j, con_loc):
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
    x, y = source

    #hacky way to make sure that we got the working grid slot..
    found_val = 0

    # otherwise...
    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1][2] > 0):
        found_connection[0] = t_grid[y][x + 1][2]
        con_loc[0] = [x + 1, y]
        found_val = found_connection[0]
    elif( (x - 1) >= 0 and t_grid[y][x - 1][2] > found_val):
        print("x - 1")
        found_connection[0] = t_grid[y][x - 1][2]
        con_loc[0] = [x - 1, y]
        found_val = found_connection[0]
    elif( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x][2] > found_val):
        print("y + 1")
        found_connection[0] = t_grid[y + 1][x][2]
        con_loc[0] = [x, y + 1]
        found_val = found_connection[0]
    elif( (y - 1) >= 0 and t_grid[y - 1][x][2] > found_val):
        print("y - 1")
        found_connection[0] = t_grid[y - 1][x][2]
        con_loc[0] = [x, y - 1]

    return con_loc, found_connection, found_connection[0] > 0

def a_backtrace(t_grid, perm_grid, sink_locations, wire_num, all_sinks):
    print("entering backtrace...")
    # new_grid = []
    # for i in range(len(perm_grid)):
    #     for j in range(len(perm_grid[0])):
    #         new_grid.append(t_grid[i][j][2])
    # new_grid = np.asarray(new_grid)
    # new_grid = np.reshape(new_grid, (len(perm_grid), len(perm_grid[0])))
    # print(new_grid)
    print(sink_locations)
    wire_value = -1 * (wire_num + 2)

    # avoid connections only between sinks - need to connect to source

    for i in range(int(len(all_sinks) / 2)):
            x_t = int(all_sinks[int(i) * 2])
            y_t = int(all_sinks[1 + int(i) * 2])
            perm_grid[y_t][x_t] = 0
    # for each sink
    x = int(sink_locations[0])
    y = int(sink_locations[1])

    print("x: " + str(x) + ", y: " + str(y))
    while True:
        # if we've reached a wire 
        if(perm_grid[y][x] == wire_value): 
            for i in range(int(len(all_sinks) / 2)):
                x_t = int(all_sinks[int(i) * 2])
                y_t = int(all_sinks[1 + int(i) * 2])
                perm_grid[y_t][x_t] = wire_value
            break
        else:
            perm_grid[y][x] = wire_value
            x, y = t_grid[y][x][:2]
            x = int(x)
            y = int(y)
    for i in range(int(len(all_sinks) / 2)):
        x_t = int(all_sinks[int(i) * 2])
        y_t = int(all_sinks[1 + int(i) * 2])
        perm_grid[y_t][x_t] = wire_value
    return perm_grid

def a_solve(perm_grid, temp_grid, wires, num_wires):
    for wire in range(num_wires):
        sinks_found = False
        wire_found = False
        source = wires[wire][1:3]
        source = [int(i) for i in source]
        all_sinks = wires[wire][3:]
        all_sinks = [int(i) for i in all_sinks]
        
        # first pin is a source
        for sink in range (int(wires[int(wire)][0]) - 1):
        # for sink in range (3):
            # used to keep track of if we have a connection between the new "path"
            # of values between the souce and the sink. Stores the numerical value
            # of the first grid location that arrives adjacent to the pin in question
            connection = np.zeros((1, 1))

            # Keeps track of the location (x, y) of each saved connection
            connection_location = np.zeros((1, 2))
            temp_grid[temp_grid > 0] = 0
            this_sink = wires[wire][3 + 2 * int(sink): 5 + 2 * int(sink)]
            this_sink = [int(i) for i in this_sink]

            d_key = a_manhattan_distance(source, this_sink)

            updated_locations = {d_key: deepcopy(this_sink)}

            print("This Sink")
            print(this_sink)
            print(source)
            print(updated_locations)

            while connection[0] == 0:
                print("updated locations")
                print(updated_locations)
                # if we have to change the value of the key and move away from the optimal path
                if not updated_locations[d_key]:
                    del updated_locations[d_key]
                if not updated_locations:
                    print("Wire " + str(sink) + " not found.")
                    break
                else:
                    d_key = min(updated_locations)


                # if not updated_locations[d_key]:
                #     del updated_locations[d_key]
                #     if not updated_locations:
                #         print("Wire " + str(sink) + " not found.")
                #         break
                #     else:
                #         

                x = updated_locations[d_key][0]
                y = updated_locations[d_key][1]

                man_dist = a_manhattan_distance(source, [x, y]) + a_manhattan_distance(this_sink, [x, y])

                temp_grid, updated_locations, wire_found, connection, connection_location = a_adjacent(temp_grid, x, y, man_dist, this_sink, source, updated_locations, perm_grid, int(wire), all_sinks, connection, connection_location)
                del updated_locations[d_key][:2]
                if(wire_found):
                    perm_grid = a_backtrace(temp_grid, perm_grid, connection_location[0], int(wire), all_sinks)
                    print(perm_grid)
                    break

                # bring the lowest manhattan distances to the front
                connection_location, connection, sinks_found = a_found_sink(temp_grid, connection, source, int(sink), connection_location)
                print("Sinks found: " + str(sinks_found))
                print("connection")
                print(connection)
                print("connection location:")
                print(connection_location[0])
                
                if(sinks_found):
                    perm_grid = a_backtrace(temp_grid, perm_grid, connection_location[0], int(wire), all_sinks)
                    print(perm_grid)

    return perm_grid