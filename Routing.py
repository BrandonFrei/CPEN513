import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import A_Star as a_star
import cv2
import os

def plot_grid(grid_n):  
    # plt.plot(grid)
    plt.imshow(grid_n, interpolation='none')
    plt.show()
    plt.clf()
    
def lm_init(input_data):
    width, height = input_data[0].split(" ")
    width = int(width)
    height = int(height)

    print("The width is " + str(width) + ", and the height is " + str(height) + ".")

    grid = np.zeros((height, width))

    # separating out the input obstructions
    num_obstructions = int(input_data[1])
    obstructions = []

    # plus 2 because of the data offset in the input file
    for i in range(2, num_obstructions+2):
        temp = input_data[i].split()
        obstructions.append(temp)

    obstructions = np.asarray(obstructions)

    wire_location = num_obstructions+2

    num_wires = int(input_data[wire_location])

    wires_temp = input_data[wire_location+1:]

    # separating out the wires
    wires = []
    for i in range(num_wires):
        temp = wires_temp[i].split()
        wires.append(temp)
        num_pins = wires[i][0]
        for j in range(int(num_pins)):
            x = int(wires[i][1 + (int)(j) * 2])
            y = int(wires[i][2 + (int)(j) * 2])
            grid[y][x] = (-1)* (i + 2)
    for i in range(num_obstructions):
        grid[int(obstructions[i][1])][int(obstructions[i][0])] = -1
    
    return wires, grid, num_wires
    
def adjacent(t_grid, x, y, value):
    """fills in values of adjacent tiles with value
       Used in the forward pass

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
    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1] == 0):
        t_grid[y][x + 1] = value + 1
        temp = [x+1, y]
        new_locations += temp
    if( (x - 1) >= 0 and t_grid[y][x - 1] == 0):
        t_grid[y][x - 1] = value + 1
        temp = [x-1, y]
        new_locations += temp
    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x] == 0):
        t_grid[y + 1][x] = value + 1
        temp = [x, y+1]
        new_locations += temp
    if( (y - 1) >= 0 and t_grid[y - 1][x] == 0):
        t_grid[y - 1][x] = value + 1
        temp = [x, y-1]
        new_locations += temp
    return np.asarray(t_grid), new_locations


def found_sink(t_grid, found_connection, x, y, j, con_loc):
    """looks for sinks in adjacent squares at the current grid tile
       Used in the forward pass

    Args:
        t_grid (numpy array): grid with the distances from the source
        found_connection (numpy array): location of the 
        x (int): x coordinate of current grid tile
        y (int): y coordinate of current grid tile
        j (int): the number of the sink we're looking at

    Returns:
        connection_location [numpy array]: location of the tile closest to the sink (if connected to net)
        found_connection [int]: value of the tile closest to the sink (if connected to net)
        boolean: if we have found all of the sinks
    """
    num_pins = len(found_connection)

    if( (x + 1) < t_grid.shape[1] and t_grid[y][x + 1] > 0):
        found_connection[j] = t_grid[y][x + 1]
        con_loc[j] = [x + 1, y]
    elif( (x - 1) >= 0 and t_grid[y][x - 1] > 0):
        found_connection[j] = t_grid[y][x - 1]
        con_loc[j] = [x - 1, y]
    elif( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x] > 0):
        found_connection[j] = t_grid[y + 1][x]
        con_loc[j] = [x, y + 1]
    elif( (y - 1) >= 0 and t_grid[y - 1][x] > 0):
        found_connection[j] = t_grid[y - 1][x]
        con_loc[j] = [x, y - 1]

    return con_loc, found_connection, (np.count_nonzero(found_connection) >= num_pins)

def backtrace(perm_grid, sink_locations_o, sink_locations, wire_num):
    """ Traces through the grid created by adjacent from the sink back to the source
        Used in the backward pass

    Args:
        temp_grid (numpy array): grid with values indicating distance from source
        perm_grid (numpy array): grid with only previous wires and blocks
        sink_locations_o (numpy array): the closest location to the sink with a value
        sink_locations (numpy array): array where the sinks are located
        wire_num (int): current wire

    Returns:
        perm_grid (numpy array): grid updated with new wire values
    """
    
    # using a value of -1 for the blocks, -2 and lower for each wire
    wire_value = -1 * (wire_num + 2)
    
    for sink in range(sink_locations_o.shape[0]):
        
        start_loc = np.asarray([int(sink_locations[int(sink) * 2]), int(sink_locations[1 + int(sink) * 2])])
        previous_location = np.asarray(deepcopy(start_loc))
        
        x = int(sink_locations_o[int(sink)][0])
        y = int(sink_locations_o[int(sink)][1])
        
        i = 0
        while True:
            cv2.imshow("img", (a_star.convert_color(perm_grid, 0)))
            cv2.waitKey(50)
            i += 1
            
            # if we've arrived at the source
            if (perm_grid[y][x] == wire_value):
                perm_grid[y][x] = wire_value
                break
            # failsafe
            if (i > 10000):
                break
            curr_val = perm_grid[y][x]
            if(curr_val == 2):
                perm_grid[y][x] = wire_value
                break

            perm_grid[y][x] = wire_value
            # check if we're near a previously placed wire
            # if we're looking at a valid grid point, if it is a net we're solving for, if we haven't previously been at the point, 
            if ( x + 1 < perm_grid.shape[1] and perm_grid[y][x + 1] == wire_value and (previous_location==np.asarray([x+1,y])).all() == False):
                previous_location = np.asarray([x, y])
                x += 1
                continue

            if ( x - 1 >= 0 and perm_grid[y][x - 1] == wire_value and (previous_location==np.asarray([x-1,y])).all() == False):
                previous_location = np.asarray([x, y])
                x -= 1
                continue
            
            if (y + 1 < perm_grid.shape[0] and perm_grid[y + 1][x] == wire_value and (previous_location==np.asarray([x,y+1])).all() == False):
                previous_location = np.asarray([x, y])
                y += 1
                continue

            if (y - 1 >= 0 and perm_grid[y - 1][x] == wire_value and (previous_location==np.asarray([x,y-1])).all() == False):
                previous_location = np.asarray([x, y])
                y -= 1
                continue

            # If we didn't find another wire, which tile do we backtrace to
            if ( x + 1 < perm_grid.shape[1] and (perm_grid[y][x + 1] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                x += 1
                continue
            if ( x - 1 >= 0 and (perm_grid[y][x - 1] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                x -= 1
                continue
            if ( y + 1 < perm_grid.shape[0] and (perm_grid[y + 1][x] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                y += 1
                continue
            if (y - 1 >= 0 and (perm_grid[y - 1][x] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                y -= 1
                continue
    # refill sink values 
    for sink in range(len(sink_locations)//2):
        x = int(sink_locations[int(sink) * 2])
        y = int(sink_locations[1 + int(sink) * 2])
        perm_grid[y][x] = wire_value

    grid[grid > 0] = 0
    return perm_grid

def lm_solve(wires, grid, num_wires):
    """Solves a net using the Lee Moore algorithm to find connections between a wire source and 
       its associated sinks.

    Args:
        wires ([type]): [description]
        grid ([type]): [description]
        num_wires ([type]): [description]

    Returns:
        [type]: [description]
    """
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', grid.shape[1] * 25, grid.shape[0] * 25)
    successful_routes = 0

    for wire in range(num_wires):
        sink_found = False
        t_value = 1

        grid[grid > 0] = 0

        updated_locations = np.asarray(wires[wire][1:3])

        # used to keep track of if we have a connection between the new "path"
        # of values between the souce and the sink. Stores the numerical value
        # of the first grid location that arrives adjacent to the pin in question
        connection = np.zeros((int((wires[wire][0])) - 1))

        # Keeps track of the location (x, y) of each saved connection
        connection_location = np.zeros((int((wires[wire][0])) - 1, 2))

        # print(updated_locations)
        while sink_found != True:
            new_locations = []
            # Looping over each "value" in the grid, one at a time, and finding the 
            # unoccupied neighbors. 
            for i in range(int(len(updated_locations)/2)):
                x = int(updated_locations[0 + i * 2])
                y = int(updated_locations[1 + i * 2])
                grid, temp_locations = adjacent(grid, x, y, t_value)
                new_locations += temp_locations
            t_value += 1
            if not new_locations:
                print("No connections found for wire number " + str(wire) + ".")
                break
            updated_locations = deepcopy(new_locations)
            
            # Did any of the new values arrive at the sinks
            # number of sinks per wire
            for j in range(int((wires[wire][0])) - 1):
                if (connection[j] > 0):
                    continue
                x = int(wires[wire][3 + int(j) * 2])
                y = int(wires[wire][4 + int(j) * 2])
                connection_location, connection, sink_found = found_sink(grid, connection, x, y, j, connection_location)
            
            if (sink_found):
                successful_routes += 1
                # delete this line when testing is done
                sink_found = True
                sinks = wires[wire][3:]
                np.savetxt("foo.txt", grid, delimiter=" ", fmt='%d')
                grid = backtrace(grid, connection_location, sinks, int(wire))
                grid[grid > 0] = 0
                # plot_grid(grid)

    position = (grid.shape[1] * 2, grid.shape[0] * 24)
    temp_img = cv2.resize(a_star.convert_color(grid, 0), (grid.shape[1] * 25, grid.shape[0] * 25), interpolation = cv2.INTER_AREA)
    cv2.putText(
        temp_img, #numpy array on which text is written
        "Wires Successfully Routed: " + str(successful_routes), #text
        position, #position at which writing has to start
        cv2.FONT_HERSHEY_COMPLEX, #font family
        grid.shape[0] / 30, #font size
        (209, 80, 0, 255), #font color
        1) #font stroke
    # destroy all, or else 2 pop up
    cv2.destroyAllWindows()
    cv2.imshow('img', temp_img)
   
    cv2.waitKey(20000)
    return grid



script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

# CHANGE FOLLOWING LINE FOR CHANGING THE INFILE
rel_path = "benchmarks/benchmarks/impossible2.infile"
abs_file_path = os.path.join(script_dir, rel_path)
# CHANGE FOLLOWING LINE TO False FOR A*
lee_moore = True

# reading in the file
data = []
with open(abs_file_path) as rfile:
    data_raw = rfile.readlines()
    for line in data_raw:
        data.append(line.strip())

if (lee_moore):
    wires, grid, num_wires = lm_init(data)
    grid = lm_solve(wires, grid, num_wires)
    np.savetxt("foo.txt", grid, delimiter=" ", fmt='%d')
else:
    perm_grid_a, temp_grid_a, wires_a, num_wires_a = a_star.a_init_aStar(data)
    perm_grid_a = perm_grid_a.astype(int)
    temp_grid_a = temp_grid_a.astype(int)
    final_grid = a_star.a_solve(perm_grid_a, temp_grid_a, wires_a, num_wires_a)



    # TODO (COMPLETED, UNTESTED) have a check for if there's a blank list being returned in new_locations.
    # in that case, need to end the current net search and draw connections that have 
    # been made, and move on to the next netlist.
    # Might be better to just delete the entire one that wasn't finished, and reschedule
    # it for later. 

    # TODO (COMPLETED, UNTESTED) Because of the case where we only have 1 wire, the solve function needs to be
    # sent a single wire at a time. Otherwise, if we only have 1 wire, will have python 
    # issues. 

    # TODO (COMPLETED, UNTESTED) make sure that we can connect to the source in all 4 directions, same with sink
