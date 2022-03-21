import numpy as np
import cv2
from queue import PriorityQueue

# set up start and goal node here
# angle search determines whether the angle goal needs to be reached or any angle is acceptable as long as the
# x and y values are in range
x_start = 20
y_start = 20
angle_start = 30

x_goal = 75
y_goal = 220
angle_goal = 30
angle_search = True

# set up clearance, radius, step size, and thresholds
# made clearance the radius plus the clearance, this can be altered if the robot wasn't meant to be that far
# from an obstacle
radius = 10
clearance = 5 + radius
step_size = 10
threshold = 0.5
angle_threshold = 30

# the weight is added to the cost value in new_cost = cost + heuristic
# in a more traditional manner it would be added to the heuristic as well but
# this works the same and I couldn't find an adequate weight for the heuristic or it would be too large
weight = .9

# graph dimensions
xdim = 400
ydim = 250

# set goal threshold and make threshold map size for the 0.5 threshold
goal_threshold = 1.5*radius
x_thresh = int(np.round(xdim/threshold))
y_thresh = int(np.round(ydim/threshold))

# define euclidean function to return distances from two sets of points
def euclidean(coords1, coords2):
    return pow((pow(coords1[0] - coords2[0], 2) + pow(coords1[1] - coords2[1], 2)), 0.5)

# class holds the equations for the obstacles
class Map:
    def __init__(self, clear):
        # for the triangular shape, used slopes and made the lines and then shifted them up and down 5 to add the
        # clearance, the shape isn't well suited for scaling with a constant clearance instead of scale factor
        # but this should work
        self.tri_1 = lambda x, y: (x-36) * (180-185) / (80-36) + 185 <= y <= (x - 36) * (210 - 185) / (115 - 36) + 185 + clear and y >= (x - 80) * (210 - 180) / (115 - 80) + 180 - clear
        self.tri_2 = lambda x, y: (x-36) * (180-185) / (80-36) + 185 >= y >= (x - 36) * (100 - 185) / (105 - 36) + 185 - clear and y <= (x - 80) * (100 - 180) / (105 - 80) + 180 + clear
        # just added clearance to radius for circle equation
        self.circle = lambda x, y: pow(y - 185, 2) + pow(x - 300, 2) <= pow(40 + clear, 2) and 260 - clear <= x <= 340 + clear
        # added clearance to side lengths for clearance for hexagon, assumed equal sides
        self.hex_1 = lambda x, y: (200 - ((70 + clear) / 2)) <= x <= 200 and (x - (200 - (70 + clear) / 2)) * (-(70 + clear) * np.tan(np.pi / 6) / 2) / ((70 + clear) / 2) + 100 - (70 + clear) * np.tan(np.pi / 6) / 2 <= y <= (x - (200 - (70 + clear) / 2)) * ((70 + clear) * np.tan(np.pi / 6) / 2) / ((70 + clear) / 2) + 100 + (70 + clear) * np.tan(np.pi / 6) / 2
        self.hex_2 = lambda x, y: 200 <= x <= (200 + (70 + clear) / 2) and (x - 200) * ((70 + clear) * np.tan(np.pi / 6) / 2) / ((70 + clear) / 2) + 100 - (70 + clear) * np.tan(np.pi / 6) <= y <= (x - 200) * (-(70 + clear) * np.tan(np.pi / 6) / 2) / ((70 + clear) / 2) + 100 + (70 + clear) * np.tan(np.pi / 6)
        # added clearance to borders
        self.quad = lambda x, y: x <= 0 + clear or y <= 0 + clear or x >= xdim - clear or y >= ydim - clear

    def is_obstacle(self, coords):
        # check if a coordinate is in an obstacle space
        return self.circle(coords[0], coords[1]) or self.quad(coords[0], coords[1]) or self.tri_1(coords[0], coords[1]) or self.tri_2(coords[0], coords[1]) or self.hex_1(coords[0], coords[1]) or self.hex_2(coords[0], coords[1])

# node class to hold coordinates, angle, parent node, and current calculated cost
class Node:
    def __init__(self, value, parent=None):
        self.coords = (value[0], value[1])
        self.value = value
        self.x = value[0]
        self.y = value[1]
        self.angle = value[2]
        self.parent = parent
        self.dist = np.inf
        self.dist_t = np.inf

    # altered eq and hash function to allow nodes to be properly implemented as dictionary key and in priority queue
    def __eq__(self, other):
        if self.value == other.value:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)

    # move functions based on step size and the given angles to move from its facing direction
    def move_0(self):
        new_angle = self.value[2] + 0
        radians = new_angle * np.pi / 180
        new_val = (self.value[0] + step_size * (np.cos(radians)), self.value[1] + step_size * (np.sin(radians)),
                   new_angle)
        return Node(new_val, self)

    def move_up30(self):
        new_angle = self.value[2] + 30
        radians = new_angle * np.pi / 180
        new_val = (self.value[0] + step_size * (np.cos(radians)), self.value[1] + step_size * (np.sin(radians)),
                   new_angle)
        return Node(new_val, self)

    def move_up60(self):
        new_angle = self.value[2] + 60
        radians = new_angle * np.pi / 180
        new_val = (self.value[0] + step_size * (np.cos(radians)), self.value[1] + step_size * (np.sin(radians)),
                   new_angle)
        return Node(new_val, self)

    def move_down30(self):
        new_angle = self.value[2] - 30
        radians = new_angle * np.pi / 180
        new_val = (self.value[0] + step_size * (np.cos(radians)), self.value[1] + step_size * (np.sin(radians)),
                   new_angle)
        return Node(new_val, self)

    def move_down60(self):
        new_angle = self.value[2] - 60
        radians = new_angle * np.pi / 180
        new_val = (self.value[0] + step_size * (np.cos(radians)), self.value[1] + step_size * (np.sin(radians)),
                   new_angle)
        return Node(new_val, self)

    # function to generate path from a node all the way to the start
    def gen_path(self):
        traceback = []
        counter = self
        while counter.parent:
            traceback.append(counter)
            counter = counter.parent
        traceback.append(counter)
        traceback.reverse()
        return traceback


# Graph class starts with empty dictionary
# key is a node, value holds list with all the associated edges
# holds obstacle map inside to check
class Graph:
    def __init__(self, map_0):
        self.graph = {}
        self.map = map_0

    # function to use to search for the surroundings for a new node and generate graph
    # adds all move function result nodes to the edges list unless they are part of an obstacle
    def gen_nodes(self, n):
        edges = [(n.move_0(), step_size), (n.move_up30(), step_size), (n.move_up60(), step_size),
                 (n.move_down30(), step_size),
                 (n.move_down60(), step_size)]

        edges = list(filter(lambda val: not self.map.is_obstacle(val[0].coords), edges))

        self.graph[n] = edges

# initialize data structures for algorithm
q = PriorityQueue()
object_space = Map(clearance)
graph = Graph(object_space)


def a_star():
    # make start node and check if start and goal are in valid space, initialize visited list
    start = Node((x_start, y_start, angle_start))
    if object_space.is_obstacle(start.coords) or object_space.is_obstacle((x_goal, y_goal, angle_goal)):
        print("Start or goal in obstacle space")
        return None

    visited = np.zeros((x_thresh, y_thresh, 1+360//angle_threshold))

    # start the priority queue with the start node
    start.dist = 0
    q.put((0, 0, start))

    # counter just kept for the priority queue as it couldn't compare nodes and needed an intermediary value
    j = 1
    # loop while the priority queue has a node and pop it out and get the distance and node
    while not q.empty():
        curr_dist, k, curr_node = q.get()
        # round numbers to find threshold value in the visited matrix
        # if the node matches the value from the visited matrix, ignore it
        # if not seen, add these values to the visited matrix
        round_x = int(np.round(curr_node.value[0]))
        round_y = int(np.round(curr_node.value[1]))
        if curr_node.value[2] < 0:
            round_angle = 360-((-1*curr_node.value[2]) % 360)
        else:
            round_angle = curr_node.value[2] % 360

        if visited[round_x, round_y, round_angle//angle_threshold] == 1:
            continue
        visited[round_x, round_y, round_angle//angle_threshold] = 1

        # return node if it is within the distance to goal node
        # angle search boolean declared at top determines if the program needs the given goal angle
        if euclidean(curr_node.value, (x_goal, y_goal)) <= goal_threshold:
            if angle_search:
                if round_angle == angle_goal:
                    return curr_node
            else:
                return curr_node

        # generate edges and nodes around the current node
        graph.gen_nodes(curr_node)

        # loop through all the adjacent nodes and check distance value
        # if lower than current, update the value in the node and add the new distance with the heuristic to the q
        # heuristic determined by euclidean distance from goal node
        # weight also added as explained earlier
        for neighbor, cost in graph.graph[curr_node]:
            heuristic = euclidean(neighbor.value, (x_goal, y_goal))
            new_dist = weight*(curr_dist + cost) + heuristic
            print(heuristic)
            print(curr_dist+cost)
            if new_dist < neighbor.dist:
                neighbor.dist = new_dist
                q.put((new_dist, j, neighbor))
                j += 1
    # return Not found if node was not found for some reason
    print("Not found")
    return None

# function called to call the algorithm and visualize it
def visual():
    goal = a_star()
    # print another statement and return None if invalid algorithm output
    if not goal:
        print("Invalid, select new nodes")
        return None

    # set up array as completely black image for now, will update later
    frame = np.zeros((y_thresh+1, x_thresh+1, 3), np.uint8)

    # add goal node threshold circle to the image
    cv2.circle(frame, (int(x_goal*2), int(y_thresh-y_goal*2)), int(goal_threshold*2), (255, 0, 0), -1)

    # create an obstacle space map with a clearance of 0 to show original obstacle sizes
    # check every block and turn white if it is an obstacle
    clear_visual = 0
    visual_space = Map(clear_visual)
    for x in range(0, x_thresh+1):
        for y in range(0, y_thresh+1):
            if visual_space.is_obstacle((x*threshold, y*threshold)):
                frame[y_thresh-y, x] = (0, 0, 255)

    # use function to generate path from start to goal
    traceback = goal.gen_path()

    # start a video file to write to
    # these settings worked for me on Ubuntu 20.04 to make a video
    output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 120, (x_thresh+1, y_thresh+1))

    # adds exploration path to video by drawing lines from node to node going through the graph dictionary
    # saves a frame every 40 nodes to save video time
    i = 0
    for node in graph.graph.keys():
        x1, y1 = node.coords
        x1 = int(np.round(2*x1))
        y1 = int(np.round(2*y1))
        frame[y_thresh-y1, x1] = (255, 255, 255)
        for pair in graph.graph[node]:
            x2, y2 = pair[0].coords
            x2 = int(np.round(2 * x2))
            y2 = int(np.round(2 * y2))
            cv2.line(frame, (x1, y_thresh-y1), (x2, y_thresh-y2), (255, 255, 255), 1)

        i += 1
        if i % 40 == 0:
            output.write(frame)

    # colors the path line at the end and adds a few seconds worth of frames
    prev = None
    for node in traceback:
        x1, y1 = node.coords
        x1 = int(np.round(2 * x1))
        y1 = int(np.round(2 * y1))
        frame[y_thresh-y1, x1] = (0, 255, 0)
        if prev:
            x2, y2 = prev.coords
            x2 = int(np.round(2 * x2))
            y2 = int(np.round(2 * y2))
            cv2.line(frame, (x1, y_thresh-y1), (x2, y_thresh-y2), (0, 255, 0), 1)
        prev = node
    for i in range(0, 480):
        output.write(frame)
    output.release()

visual()
