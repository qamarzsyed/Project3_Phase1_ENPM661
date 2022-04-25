ENPM 661 Project 3 Phase 1 ReadMe file - Qamar Syed

Environment: Python 3.7 with Numpy, PriorityQueue, and OpenCV libraries

Steps to run:

1. EDIT PARAMETERS: 

The parameters are stored at the top of the qamar_syed.py with the names being the name of the appropriate parameter

There is a boolean called angle_search, this determines if the search requires the goal node to have the set angle_goal or whether any angle is acceptable 

The clearance parameter is defined as the clearance plus the radius, resulting in a distance of clearance + radius for the node from the obstacle

2. RUN SCRIPT:

Can be run from the terminal simply by entering 'python3 qamar_syed.py'

If nodes are in an incorrect location or there is any other reason the program cannot find a path between the two nodes, a message will be printed out to the terminal

Algorithm is weighted as explained in the comments starting line 26 to save runtime

3. OUTPUT VIDEO STORED IN THE DIRECTORY OF THE .PY FILE:

'output.avi' video displays result of search

Sample output video stored in the submitted zip file


