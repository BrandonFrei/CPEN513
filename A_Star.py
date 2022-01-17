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

def a_adjacent(t_grid, x, y, value, sink, source, loc_dict, perm_grid, g_v):
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


    # if (g_v == 60):
    #     new_grid = []
    #     for i in range(9):
    #         for j in range(12):
    #             new_grid.append(t_grid[i][j][2])
    #     new_grid = np.asarray(new_grid)
    #     new_grid = np.reshape(new_grid, (9, 12))
    #     print(new_grid)
    #     print(loc_dict)
    #     exit(0)
    return np.asarray(t_grid), loc_dict


# shouldn't need to modify this one too much. might need to check for the lowest value one,
# but I don't think so.
def a_found_sink(t_grid, found_connection, this_sink, j, con_loc):
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
    x, y = this_sink

    num_pins = len(found_connection)

    #hacky way to make sure that we got the working grid slot..
    found_val = 0
    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1][2] > 0):
        found_connection[j] = t_grid[y][x + 1][2]
        con_loc[j] = [x + 1, y]
        found_val = found_connection[j]
    if( (x - 1) >= 0 and t_grid[y][x - 1][2] > found_val):
        print("x - 1")
        found_connection[j] = t_grid[y][x - 1][2]
        con_loc[j] = [x - 1, y]
        found_val = found_connection[j]
    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x][2] > found_val):
        print("y + 1")
        found_connection[j] = t_grid[y + 1][x][2]
        con_loc[j] = [x, y + 1]
        found_val = found_connection[j]
    if( (y - 1) >= 0 and t_grid[y - 1][x][2] > found_val):
        print("y - 1")
        found_connection[j] = t_grid[y - 1][x][2]
        con_loc[j] = [x, y - 1]

    return con_loc, found_connection, (np.count_nonzero(found_connection) >= num_pins)

def a_backtrace(t_grid, perm_grid, sink_locations, wire_num):
    print("entering backtrace...")
    new_grid = []
    for i in range(len(perm_grid)):
        for j in range(len(perm_grid[0])):
            new_grid.append(t_grid[i][j][2])
    new_grid = np.asarray(new_grid)
    new_grid = np.reshape(new_grid, (len(perm_grid), len(perm_grid[0])))
    print(new_grid)
    np.savetxt("foo.txt", new_grid, delimiter=" ", fmt='%d')
    print(sink_locations)
    wire_value = -1 * (wire_num + 2)

    # avoid connections only between sinks - need to connect to source

    # for each sink
    x = int(sink_locations[0])
    y = int(sink_locations[1])
    
    while True:
        # if we've reached a wire that is not the original source
        if(perm_grid[y][x] == wire_value): 
            print("broke here")
            break
        else:
            perm_grid[y][x] = wire_value
            x, y = t_grid[y][x][:2]
            x = int(x)
            y = int(y)
            #print("x: " + str(x) + ", y: " + str(y))
            # print(t_grid[y][x])
            # exit(0)

    # change the sink values back to their original values

    # exit(0)
    #TODO probably in main, fix multiple sinks and multiple wires
    plot_grid(perm_grid)
    return perm_grid



def a_solve(perm_grid, temp_grid, wires, num_wires):
    for wire in range(num_wires):
        global_var = 0
        print("i've gone back to the top")
        sinks_found = False

        source = wires[wire][1:3]
        source = [int(i) for i in source]

        # used to keep track of if we have a connection between the new "path"
        # of values between the souce and the sink. Stores the numerical value
        # of the first grid location that arrives adjacent to the pin in question
        connection = np.zeros((int((wires[wire][0])) - 1))

        # Keeps track of the location (x, y) of each saved connection
        connection_location = np.zeros((int((wires[wire][0])) - 1, 2))
        # first pin is a source
        for sink in range (int(wires[int(wire)][0]) - 1):
            temp_grid[temp_grid > 0] = 0

            d_key = a_manhattan_distance(source, wires[wire][3 + 2 * int(sink):5 + 2 * int(sink)])

            updated_locations = {d_key: deepcopy(source)}

            p = 0
            this_sink = wires[wire][3 + 2 * int(sink): 5 + 2 * int(sink)]
            this_sink = [int(i) for i in this_sink]
            print("This Sink")
            print(this_sink)
            print(source)
            # print("Number of Sinks")
            # print(int(wires[int(wire)][0]) - 1)
            while connection[int(sink)] == 0:
                
                # if we have to change the value of the key and move away from the optimal
                # path

                if not updated_locations[d_key]:
                    del updated_locations[d_key]
                    if not updated_locations:
                        print("Wire " + str(sink) + " not found.")
                        break
                    else:
                        d_key = min(updated_locations, key=updated_locations.get)
                # print(updated_locations)
                x = updated_locations[d_key][0]
                y = updated_locations[d_key][1]

                man_dist = a_manhattan_distance(source, [x, y]) + a_manhattan_distance(this_sink, [x, y])

                temp_grid, updated_locations = a_adjacent(temp_grid, x, y, man_dist, this_sink, source, updated_locations, perm_grid, global_var)
                global_var += 1
                del updated_locations[d_key][:2]

                # bring the lowest manhattan distances to the front
                connection_location, connection, sinks_found = a_found_sink(temp_grid, connection, this_sink, int(sink), connection_location)
                if(connection.all() > 0):
                    new_grid = []
                    for i in range(9):
                        for j in range(12):
                            new_grid.append(temp_grid[i][j][2])
                    new_grid = np.asarray(new_grid)
                    new_grid = np.reshape(new_grid, (9, 12))
                    print(new_grid)

                # print("conenction_location:")
                # print(connection_location)
                if(sinks_found):
                    break
                # this might not be necessary - I think the updated locations will catch it
                # if(man_dist > perm_grid.shape[0] + perm_grid.shape[1]):
                #     print("Wire " + str(sink) + " not found.")
                #     break
            perm_grid = a_backtrace(temp_grid, perm_grid, connection_location[int(sink)], int(wire))
    return perm_grid