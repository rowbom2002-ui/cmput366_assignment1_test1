import heapq
import math

class State:
    """
    Class to represent a state on grid-based pathfinding problems. The class contains two static variables:
    map_width and map_height containing the width and height of the map. Although these variables are properties
    of the map and not of the state, they are used to compute the hash value of the state, which is used
    in the CLOSED list. 

    Each state has the values of x, y, g, h, and cost. The cost is used as the criterion for sorting the nodes
    in the OPEN list for both Dijkstra's algorithm and A*. For Dijkstra the cost should be the g-value, while
    for A* the cost should be the f-value of the node. 
    """
    map_width = 0
    map_height = 0
    
    def __init__(self, x, y):
        """
        Constructor - requires the values of x and y of the state. All the other variables are
        initialized with the value of 0.
        """
        self._x = x
        self._y = y
        self._g = 0
        self._cost = 0
        self._parent = None
        
    def __repr__(self):
        """
        This method is invoked when we call a print instruction with a state. It will print [x, y],
        where x and y are the coordinates of the state on the map. 
        """
        state_str = "[" + str(self._x) + ", " + str(self._y) + "]"
        return state_str
    
    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._cost < other._cost
    
    def state_hash(self):
        """
        Given a state (x, y), this method returns the value of x * map_width + y. This is a perfect 
        hash function for the problem (i.e., no two states will have the same hash value). This function
        is used to implement the CLOSED list of the algorithms. 
        """
        return self._y * State.map_width + self._x
    
    def __eq__(self, other):
        """
        Method that is invoked if we use the operator == for states. It returns True if self and other
        represent the same state; it returns False otherwise. 
        """
        return self._x == other._x and self._y == other._y

    def get_x(self):
        """
        Returns the x coordinate of the state
        """
        return self._x
    
    def set_parent(self, parent):
        """
        Sets the parent of a node in the search tree
        """
        self._parent = parent

    def get_parent(self):
        """
        Returns the parent of a node in the search tree
        """
        return self._parent
    
    def get_y(self):
        """
        Returns the y coordinate of the state
        """
        return self._y
    
    def get_g(self):
        """
        Returns the g-value of the state
        """
        return self._g
        
    def set_g(self, g):
        """
        Sets the g-value of the state
        """
        self._g = g

    def get_cost(self):
        """
        Returns the cost of a state; the cost is determined by the search algorithm
        """
        return self._cost
    
    def set_cost(self, cost):
        """
        Sets the cost of the state; the cost is determined by the search algorithm 
        """
        self._cost = cost


def reconstruct_path(goal_state):
    path = []
    current = goal_state
    while current is not None:
        path.append(current)
        current = current.get_parent()
    path.reverse()
    return path


def dijkstra(start, goal, gridded_map):
    open_list = []
    closed = {}
    best_g = {}

    start.set_g(0)
    start.set_cost(0)
    start.set_parent(None)
    heapq.heappush(open_list, start)
    best_g[start.state_hash()] = 0

    while open_list:
        current = heapq.heappop(open_list)
        current_hash = current.state_hash()
        if current_hash in closed:
            continue

        closed[current_hash] = current

        if current == goal:
            return reconstruct_path(current), current.get_g(), len(closed)

        for child in gridded_map.successors(current):
            child_hash = child.state_hash()
            if child_hash in closed:
                continue
            child_g = child.get_g()
            if child_hash not in best_g or child_g < best_g[child_hash]:
                best_g[child_hash] = child_g
                child.set_parent(current)
                child.set_cost(child_g)
                heapq.heappush(open_list, child)

    return None, -1, len(closed)


def heuristic(state, goal):
    dx = abs(state.get_x() - goal.get_x())
    dy = abs(state.get_y() - goal.get_y())
    diagonal_steps = min(dx, dy)
    straight_steps = abs(dx - dy)
    return 1.5 * diagonal_steps + straight_steps


def astar(start, goal, gridded_map):
    open_list = []
    closed = {}
    best_g = {}

    start.set_g(0)
    start.set_cost(heuristic(start, goal))
    start.set_parent(None)
    heapq.heappush(open_list, start)
    best_g[start.state_hash()] = 0

    while open_list:
        current = heapq.heappop(open_list)
        current_hash = current.state_hash()
        if current_hash in closed:
            continue

        closed[current_hash] = current

        if current == goal:
            return reconstruct_path(current), current.get_g(), len(closed)

        for child in gridded_map.successors(current):
            child_hash = child.state_hash()
            if child_hash in closed:
                continue
            child_g = child.get_g()
            if child_hash not in best_g or child_g < best_g[child_hash]:
                best_g[child_hash] = child_g
                child.set_parent(current)
                child.set_cost(child_g + heuristic(child, goal))
                heapq.heappush(open_list, child)

    return None, -1, len(closed)
    
