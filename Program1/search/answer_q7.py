from typing import Literal, List, Tuple, TypeAlias, Annotated

import numpy as np

State: TypeAlias = Tuple[int, int, str]

class Node:
    def __init__(self, id: int, state: State, pNode: 'Node'=None, pAct: str='', level: int=0, cost: float=0, utility=0):
        self.id = id
        self.state = state
        self.pNode = pNode
        self.pAct = pAct
        self.level = level
        self.cost = cost
        self.utility = utility

#===============================================================================
# 7.1 FORMULATION
#===============================================================================

def state_func(grid: np.ndarray) -> State:
    """Return a state based on the grid (observation).

    Number mapping:
    -  0: dirt (passable)
    -  1: wall (not passable)
    -  2x: agent is facing up (north)
    -  3x: agent is facing right (east)
    -  4x: agent is facing down (south)
    -  5x: agent is facing left (west)
    -  6: goal
    -  7: mud (passable, but cost more)
    -  8: grass (passable, but cost more)

    State is a tuple of
    - x (int)
    - y (int)
    - facing ('N', 'E', 'S', or 'W')
    """
    # TODO
    state: State = None
    orientation = ['N', 'E', 'S', 'W'] # index: 0, 1, 2, 3, 4
    start_facing_n = grid[1,1]

    for i in range(len(orientation)):
        if (start_facing_n // 10 == i + 2):
            state = (1, 1, orientation[i])
            break
    return state

# TODO
# R = Turn Right
# L = Turn Left
# F = Move Forward
ACTIONS: List[str] = ['R', 'L', 'F'] 

def transition(state: State, actoin: str, grid: np.ndarray) -> State:
    """Return a new state."""
    # TODO
    new_state: State = state
    x, y, cur_facing = state 
    orientation = ['N', 'E', 'S', 'W']

    if (actoin == 'R'):
        for i in range(len(orientation)):
            if (orientation[i] == cur_facing):
                new_state = (x, y, orientation[(i+1) % 4])
                break
    elif (actoin == 'L'):
        for i in range(len(orientation)):
            if (orientation[i] == cur_facing):
                new_state = (x, y, orientation[(i+3) % 4])
                break
    elif (actoin == 'F'):
        if (cur_facing == 'N' and grid[y-1, x] != 1):
            new_state = (x, y-1, cur_facing)
        elif (cur_facing == 'E' and grid[y, x+1] != 1):
            new_state = (x+1, y, cur_facing)
        elif (cur_facing == 'S' and grid[y+1, x] != 1):
            new_state = (x, y+1, cur_facing)
        elif (cur_facing == 'W' and grid[y, x-1] != 1):
            new_state = (x-1, y, cur_facing)
    return new_state


def is_goal(state: State, grid: np.ndarray) -> bool:
    """Return whether the state is a goal state."""
    # TODO
    x, y, f = state
    m, n = grid.shape
    return (x == m - 2) and (y == n - 2)


def cost(state: State, actoin: str, grid: np.ndarray) -> float:
    """Return a cost of an action on the state."""
    # TODO
    x, y, f = transition(state, actoin, grid)
    return grid[y, x]


#===============================================================================
# 7.2 SEARCH
#===============================================================================


def heuristic(state: State, goal_state: State) -> float:
    """Return the heuristic value of the state."""
    # TODO
    x, y, f = state
    xg, yg, fg = goal_state
    # Manhattan distance
    return abs(xg - x) + abs(yg - y)

# solution_plan return action_plan and state_plan
def solution_plan(node: 'Node'):

    state_plan = [node.state]
    action_plan = []
    
    while (node.pNode != None):
        action_plan.insert(0, node.pAct)
        node = node.pNode
        state_plan.insert(0, node.state)
    return (action_plan, state_plan)
#end solution_plan

# utility function
# Return
# The higher the number, the higher the utility
# The lower the number, the lower the utility
def utility(strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'], node: Node, goal_state: State):
    if (strategy == 'DFS'):
        return node.level
    elif (strategy == 'BFS'):
        return -node.level
    elif (strategy == 'UCS'):
        return -node.cost
    elif (strategy == 'GS'):
        return -heuristic(node.state, goal_state)
    elif (strategy == 'A*'):
        return -node.cost - heuristic(node.state, goal_state)
#end utility

# Add node to Frontier (a list that sort according to utility of the node)
def add_to_frontier(frontier: list(), new_node: Node):
    frontier.append(new_node)
    # Sort from higher to lower utility 
    frontier.sort(key=lambda node: node.utility, reverse=True)
#end add_to_frontier

# Get the node in frontier that contains the state if exists
def get_match_state(frontier: list(), new_state: State):
    for node in frontier:
        if node.state == new_state:
            return node
    return None
#end get_match_state

# Remove old node with id r_id and add new node (that has higher utility value) 
def replace(frontier: list(), r_id: int, new_node: Node):
    for node in frontier:
        if node.id == r_id:
            frontier.remove(node)
            break
    add_to_frontier(frontier, new_node)
#end replace

def graph_search(
        grid: np.ndarray,
        strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
        ) -> Tuple[
            Annotated[List[str], 'actions of the plan'],
            Annotated[List[State], 'states of the plan'],
            Annotated[List[State], 'explored states']]:
    """Return a plan (actions and states) and a list of explored states (in order)."""

    # TODO
    # Get initial state from env
    init_state = state_func(grid)
    m, n = grid.shape
    goal_state: State = (n-2, m-2, '')

    # Initialize start node
    id = 0
    init_node = Node(id, init_state)
    init_node.utility = utility(strategy, init_node, goal_state)

    # Initialize frontier (as list)
    frontier = list()
    add_to_frontier(frontier, init_node)

    # Initialize explored_set (as dict)
    explored_set = dict()
    
    # If frontier is empty, return failure
    while (len(frontier) != 0):
        # Choose a leaf node and remove it from the frontier (one with highest utility)
        node = frontier.pop(0)
        
        # If the node contains a goal state, then return solution
        if (is_goal(node.state, grid)):
            action_plan, state_plan = solution_plan(node)
            return (action_plan, state_plan, [e for e in explored_set]) # solution
        
        # Add the node state to the explored set
        explored_set[node.state] = 0

        # Expand the chosen node
        for A in ACTIONS:
            result_state = transition(node.state, A, grid)
            
            # if result_state not in explored_set 
            if (result_state in explored_set):
                continue # Do not add and continue

            # Add resulting node to frontier*
            id += 1
            result_node = Node(id, result_state, pNode=node, pAct=A, level=(node.level + 1), cost=(node.cost + cost(node.state, A, grid)))
            result_node.utility = utility(strategy, node, goal_state)
            
            match_node = get_match_state(frontier, result_state)

            # *if result_node state is not matched with any node state in frontier
            if (match_node == None):
                add_to_frontier(frontier, result_node)
            elif ((strategy == 'GS' or strategy == 'A*') and match_node.utility < result_node.utility):
                replace(frontier, match_node.id, result_node)
        #end for 
    #end while
    return None 
