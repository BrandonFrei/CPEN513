from turtle import color
import numpy as np
from copy import deepcopy
import time
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import colors

from numpy.lib.utils import source


def plot_grid(grid_n):
    # plt.plot(grid)
    plt.imshow(grid, interpolation='none')
    plt.show()
    plt.clf()
    

# reading in the file
data = []
with open('C:/Users/flyer/OneDrive/Documents/UBC School/CPEN513/CPEN513/benchmarks/benchmarks/wavy.infile') as rfile:
    data_raw = rfile.readlines()
    for line in data_raw:
        data.append(line.strip())
        
width, height = data[0].split(" ")
width = int(width)
height = int(height)

print("The width is " + str(width) + ", and the height is " + str(height) + ".")

grid = np.zeros((height, width))

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
        grid[y][x] = (-1)* (i + 2)
for i in range(num_obstructions):
    grid[int(obstructions[i][1])][int(obstructions[i][0])] = -1
    
def adjacent(t_grid, x, y, value):
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


def found_sink(t_grid, found_connection, x, y, j):
    """looks for sinks in adjacent squares at the current grid tile

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
        connection_location[j] = [x + 1, y]
    if( (x - 1) >= 0 and t_grid[y][x - 1] > 0):
        found_connection[j] = t_grid[y][x - 1]
        connection_location[j] = [x - 1, y]
    if( (y + 1) < t_grid.shape[0] and t_grid[y + 1][x] > 0):
        found_connection[j] = t_grid[y + 1][x]
        connection_location[j] = [x, y + 1]
    if( (y - 1) >= 0 and t_grid[y - 1][x] > 0):
        found_connection[j] = t_grid[y - 1][x]
        connection_location[j] = [x, y - 1]

    return connection_location, found_connection, (np.count_nonzero(found_connection) >= num_pins)

# temp_grid -> the grid with the values filled in from the lee-moore algorithm
# perm_grid -> the grid with the current connections / blocks only
# x, y -> the starting point of the back trace algorithm (the grid slot closest
#      to the sink in question)
# sink_locations_o -> offset sink locations, where the first value is located
# sink_locations -> where the sink is located
def backtrace(temp_grid, perm_grid, sink_locations_o, sink_locations, wire_num):
    # using a value of -1 for the blocks, -2 and lower for each wire
    """ Traces through the grid created by adjacent from the sink back to the source

    Args:
        temp_grid (numpy array): grid with values indicating distance from source
        perm_grid (numpy array): grid with only previous wires and blocks
        sink_locations_o (numpy array): the closest location to the sink with a value
        sink_locations (numpy array): array where the sinks are located
        wire_num (int): current wire

    Returns:
        perm_grid (numpy array): grid updated with new wire values
    """

    wire_value = -1 * (wire_num + 2)
    for sink in range(sink_locations_o.shape[0]):
        
        start_loc = np.asarray([int(sink_locations[int(sink) * 2]), int(sink_locations[1 + int(sink) * 2])])
        previous_location = np.asarray(deepcopy(start_loc))
        
        x = int(sink_locations_o[int(sink)][0])
        y = int(sink_locations_o[int(sink)][1])
        
        i = 0
        while True:
            i += 1
            perm_grid[y][x] = wire_value
            
            # if we've arrived at the source
            if (temp_grid[y][x] == wire_value):
                break
            # failsafe
            if (i > 10000):
                break
            curr_val = temp_grid[y][x]
            if(curr_val == 2):
                break

            # if we're looking at a valid grid point
            if ( x + 1 < temp_grid.shape[1] and temp_grid[y][x + 1] == wire_value and not (previous_location==start_loc).all()):
                previous_location = np.asarray([x, y])
                x += 1
                continue

            if ( x - 1 >= 0 and temp_grid[y][x - 1] == wire_value and not (previous_location==start_loc).all()):
                previous_location = np.asarray([x, y])
                x -= 1
                continue
            
            if (y + 1 < temp_grid.shape[0] and temp_grid[y + 1][x] == wire_value and not (previous_location==start_loc).all()):
                previous_location = np.asarray([x, y])
                y += 1
                continue

            if (y - 1 >= 0 and temp_grid[y - 1][x] == wire_value and not (previous_location==start_loc).all()):
                previous_location = np.asarray([x, y])
                y -= 1
                continue

            # which way do we backtrack, or if we've found a net connecting another source
            # and sink on the same network, we can stop there. Must make sure not to go
            # back to the previous tile, though.
            if ( x + 1 < temp_grid.shape[1] and (temp_grid[y][x + 1] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                x += 1
                continue
            if ( x - 1 >= 0 and (temp_grid[y][x - 1] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                x -= 1
                continue
            if ( y + 1 < temp_grid.shape[0] and (temp_grid[y + 1][x] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                y += 1
                continue
            if (y - 1 >= 0 and (temp_grid[y - 1][x] == curr_val - 1 )):
                previous_location = np.asarray([x, y])
                y -= 1
                continue

    return perm_grid               

for wire in range(num_wires):
    temp_values = np.asarray(wires[wire][1:3])
    sink_found = False
    t_value = 1
    # print("updated grid")
    # print(grid)
    updated_grid = np.asarray(deepcopy(grid))
    updated_locations = deepcopy(temp_values)

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
            updated_grid, temp_locations = adjacent(updated_grid, x, y, t_value)
            new_locations += temp_locations
        t_value += 1
        if not new_locations:
            print("No connections found for wire number " + str(wire) + ".")
            break
        updated_locations = deepcopy(new_locations)
        
        # Did any of the new values arrive at the sinks
        # if (t_value == 16):
        # for each wire
        # number of sinks per wire
        for j in range(int((wires[wire][0])) - 1):
            if (connection[j] > 0):
                continue
            x = int(wires[wire][3 + int(j) * 2])
            y = int(wires[wire][4 + int(j) * 2])
            connection_location, connection, sink_found = found_sink(updated_grid, connection, x, y, j)
            # print("connection")
            # print(connection)
        if (sink_found or t_value > 1000):
        #if (sink_found):

            # delete this line when testing is done
            sink_found = True
            sinks = wires[wire][3:]
            np.savetxt("foo.txt", updated_grid, delimiter=" ", fmt='%d')
            grid = backtrace(updated_grid, grid, connection_location, sinks, int(wire))
            plot_grid(grid)


grid = grid.astype(int) * -1
print(grid)
# np.savetxt("foo.txt", grid, delimiter=" ", fmt='%d')
color_grid = np.asarray([["white"]*grid.shape[1]]*grid.shape[0])

color_grid[grid == -1] = "black"
# plt.imshow(grid, interpolation='none')
# plt.show()
# colors = ["yellow", "cyan", "purple", "green", "red", "orange", "pink"]
# for i in range(2, 2 + int(num_wires)):
#     color_grid[grid == -i] = colors[i]

# print(color_grid)






    # TODO (COMPLETED, UNTESTED) have a check for if there's a blank list being returned in new_locations.
    # in that case, need to end the current net search and draw connections that have 
    # been made, and move on to the next netlist.
    # Might be better to just delete the entire one that wasn't finished, and reschedule
    # it for later. 

    # TODO (COMPLETED, UNTESTED) Because of the case where we only have 1 wire, the solve function needs to be
    # sent a single wire at a time. Otherwise, if we only have 1 wire, will have python 
    # issues. 

    # TODO (COMPLETED, UNTESTED) make sure that we can connect to the source in all 4 directions, same with sink
