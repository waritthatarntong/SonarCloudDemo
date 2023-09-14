# from typing import Literal, List, Tuple, TypeAlias, Annotated

# import numpy as np

# State: TypeAlias = Tuple[int, int, str]

# class Node:
#     def __init__(self, id: int, state: State, pNode: 'Node'=None, pAct: str='', level: int=0, cost: float=0, priority=0):
#         self.id = id
#         self.state = state
#         self.pNode = pNode
#         self.pAct = pAct
#         self.level = level
#         self.cost = cost
#         self.priority = priority

# #===============================================================================
# # 7.1 FORMULATION
# #===============================================================================

# def state_func(grid: np.ndarray) -> State:
#     """Return a state based on the grid (observation).

#     Number mapping:
#     -  0: dirt (passable)
#     -  1: wall (not passable)
#     -  2x: agent is facing up (north)
#     -  3x: agent is facing right (east)
#     -  4x: agent is facing down (south)
#     -  5x: agent is facing left (west)
#     -  6: goal
#     -  7: mud (passable, but cost more)
#     -  8: grass (passable, but cost more)

#     State is a tuple of
#     - x (int)
#     - y (int)
#     - facing ('N', 'E', 'S', or 'W')
#     """
#     # TODO
#     state: State = None
#     orientation = ['N', 'E', 'S', 'W'] # index: 0, 1, 2, 3, 4
#     start_facing_n = grid[1,1]

#     for i in range(len(orientation)):
#         if (start_facing_n // 10 == i + 2):
#             state = (1, 1, orientation[i])
#             break
#     return state

# # TODO
# # R = Turn Right
# # L = Turn Left
# # F = Move Forward
# ACTIONS: List[str] = ['R', 'L', 'F'] 

# def transition(state: State, actoin: str, grid: np.ndarray) -> State:
#     """Return a new state."""
#     # TODO
#     new_state: State = state
#     x, y, cur_facing = state 
#     orientation = ['N', 'E', 'S', 'W']

#     if (actoin == 'R'):
#         for i in range(len(orientation)):
#             if (orientation[i] == cur_facing):
#                 new_state = (x, y, orientation[(i+1) % 4])
#                 break
#     elif (actoin == 'L'):
#         for i in range(len(orientation)):
#             if (orientation[i] == cur_facing):
#                 new_state = (x, y, orientation[(i+3) % 4])
#                 break
#     elif (actoin == 'F'):
#         if (cur_facing == 'N' and grid[y-1, x] != 0):
#             new_state = (x, y-1, cur_facing)
#         elif (cur_facing == 'E' and grid[y, x+1] != 0):
#             new_state = (x+1, y, cur_facing)
#         elif (cur_facing == 'S' and grid[y+1, x] != 0):
#             new_state = (x, y+1, cur_facing)
#         elif (cur_facing == 'W' and grid[y, x-1] != 0):
#             new_state = (x-1, y, cur_facing)
#     return new_state


# def is_goal(state: State, grid: np.ndarray) -> bool:
#     """Return whether the state is a goal state."""
#     # TODO
#     x, y, f = state
#     m, n = grid.shape
#     return (x == m - 2) and (y == n - 2)


# def cost(state: State, actoin: str, grid: np.ndarray) -> float:
#     """Return a cost of an action on the state."""
#     # TODO
#     # Place the following lines with your own implementation
#     x, y, f = transition(state, actoin, grid)
#     return grid[y, x]


# #===============================================================================
# # 7.2 SEARCH
# #===============================================================================


# def heuristic(state: State, goal_state: State) -> float:
#     """Return the heuristic value of the state."""
#     # TODO
#     x, y, f = state
#     xg, yg, fg = goal_state
#     # Manhattan distance
#     return abs(xg - x) + abs(yg - y)

# # solution_plan return action_plan and state_plan
# def solution_plan(node: 'Node'):

#     state_plan = [node.state]
#     action_plan = []
    
#     while (node.pNode != None):
#         action_plan.insert(0, node.pAct)
#         node = node.pNode
#         state_plan.insert(0, node.state)
#     return (action_plan, state_plan)
# #end solution_plan

# # priority function
# # Return
# # The higher, the number means the higher, the priority input node has
# # The lower, the number means the lower, the priority input node has
# def priority(strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'], node: Node, goal_state: State):
#     if (strategy == 'DFS'):
#         return node.level
#     elif (strategy == 'BFS'):
#         return -node.level
#     elif (strategy == 'UCS'):
#         return -node.cost
#     elif (strategy == 'GS'):
#         return -heuristic(node.state, goal_state)
#     elif (strategy == 'A*'):
#         return -node.cost - heuristic(node.state, goal_state)
# #end priority

# # Add node to Frontier (a list that sort according to priority of the node)
# def add_to_frontier(frontier: list(), new_node: Node):
#     frontier.append(new_node)
#     # Sort from higher to lower priority 
#     frontier.sort(key=lambda node: node.priority, reverse=True)
# #end add_to_frontier

# # Get the node that contain the same state as the input node
# def get_match_state(frontier: list(), new_node: Node):
#     for node in frontier:
#         if node.state == new_node.state:
#             return node
#     return None
# #end get_match_state

# # Remove old node with id r_id and add new node (that has higher priority) 
# def replace(frontier: list(), r_id: int, new_node: Node):
#     for node in frontier:
#         if node.id == r_id:
#             frontier.remove(node)
#             break
#     add_to_frontier(frontier, new_node)
# #end replace

# def cost_map(grid: np.ndarray) -> np.ndarray:

#     cost_grid = np.copy(grid)
#     cost_grid[1,1] = cost_grid[1,1] % 10
#     # grid map == 1 means not passable
#     cost_grid = np.where(cost_grid == 1, np.nan, cost_grid)
    
#     # grid map == 0 means dirt 
#     # we changed this to 0 so that A* behave more appropriately 
#     # as it relies on heuristic function 
#     cost_grid[cost_grid == 0] = 1

#     # grid map == 7 means mud
#     cost_grid[cost_grid == 7] = 10

#     # grid map == 8 means grass
#     cost_grid[cost_grid == 8] = 6

#     # Replace np.nan to 0
#     np.nan_to_num(cost_grid, copy=False)
#     return cost_grid
# #end cost_map

# def graph_search(
#         grid: np.ndarray,
#         strategy: Literal['DFS', 'BFS', 'UCS', 'GS', 'A*'] = 'A*'
#         ) -> Tuple[
#             Annotated[List[str], 'actions of the plan'],
#             Annotated[List[State], 'states of the plan'],
#             Annotated[List[State], 'explored states']]:
#     """Return a plan (actions and states) and a list of explored states (in order)."""

#     # TODO
#     # Get initial state from env
#     init_state = state_func(grid)
#     cost_grid = cost_map(grid)
#     m, n = grid.shape
#     goal_state: State = (n-2, m-2, '')

#     # Initialize start node
#     id = 0
#     init_node = Node(id, init_state)
#     init_node.priority = priority(strategy, init_node, goal_state)

#     # Initialize frontier (as list)
#     frontier = list()
#     add_to_frontier(frontier, init_node)

#     # Initialize explored_set (as dict)
#     explored_set = dict()
    
#     # If frontier is empty, return failure
#     while (len(frontier) != 0):
#         # Choose a leaf node and remove it from the frontier (one with highest priority)
#         node = frontier.pop(0)
        
#         # If the node contains a goal state, then return solution
#         if (is_goal(node.state, grid)):
#             action_plan, state_plan = solution_plan(node)
#             return (action_plan, state_plan, [e for e in explored_set]) # solution
        
#         # Add the node state to the explored set
#         explored_set[node.state] = 0

#         # Expand the chosen node
#         for A in ACTIONS:
#             result_state = transition(node.state, A, cost_grid)
#             id += 1
#             result_node = Node(id, result_state, pNode=node, pAct=A, level=(node.level + 1), cost=(node.cost + cost(node.state, A, cost_grid)))
#             result_node.priority = priority(strategy, node, goal_state)
            
#             # Add resulting node to frontier
#             # if result_state not in explored_set 
#             if (result_state in explored_set):
#                 continue # Do not add and continue
            
#             match_node = get_match_state(frontier, result_node)

#             # if result_node state is not matched with any node state in frontier
#             if (match_node == None):
#                 add_to_frontier(frontier, result_node)
#             elif ((strategy == 'GS' or strategy == 'A*') and match_node.priority < result_node.priority):
#                 replace(frontier, match_node.id, result_node)
#         #end for 
#     #end while
#     return None 

# import env

# # During the development you can reduce this GRID_SIZE
# GRID_SIZE = 6
# CELL_SIZE = 20
# WINDOW_SIZE = (CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE)

# grid = env.gen_maze(GRID_SIZE, add_mud=True, add_grass=True)

# plan_actions, plan_states, explored_states = graph_search(grid, 'A*')

# print('The plan:', ', '.join(plan_actions))

# overlay = np.zeros_like(grid)
# total_cost = 0.0
# for action, state in zip(plan_actions, plan_states):
#     total_cost += cost(state, action, grid)

# print('The plan cost:', total_cost)
# print('Total explored states (iterations):', len(explored_states))

# x = 9
# print(-x)

# state = (1,1, 'B')
# d = {(1, 1, 'A'): 1, (1, 1, 'B'): 1}
# print(d)
# if (state not in d):
#     d[state] = 1
# else:
#     d[state] = d[state] + 1
# print(d)

# strategy = 'BFS'
# test = (strategy == 'GS' or strategy == 'A*') and True
# print(test)