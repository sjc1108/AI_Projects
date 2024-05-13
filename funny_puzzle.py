import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0

    for i, tile in enumerate(from_state):
        if tile != 0:
            from_x = i % 3  #col of nnum in curr
            from_y = i // 3  #row
            goal_i = to_state.index(tile) #for index of curr tile
            to_x = goal_i % 3  #col of num in goal
            to_y = goal_i // 3  #row
            distance += abs(from_x - to_x) + abs(from_y - to_y)


    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))



def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    possible_dir = [(-1, 0), (1, 0), (0, -1), (0, 1)] #up down left right
    succ = []
    empty_i = [i for i, val in enumerate(state) if val == 0] #empty indices 

    for slot in empty_i:
        row = slot // 3 #Calc row of empty tile 
        col = slot % 3

        for drow, dcol in possible_dir:
            new_r, new_c = row + drow, col + dcol #calc new pos

            if 0 <= new_r < 3 and 0 <= new_c < 3:
                new_state = state[:] #create copy of current state for mod
                new_state[slot], new_state[new_r * 3 + new_c] = new_state[new_r * 3 +new_c], new_state[slot] #swapping
                if new_state != state: #new state vs curr state
                    succ.append(new_state) #new state to succ if diff

    return sorted(succ)
    
def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    #https://docs.python.org/3/library/heapq.html
    #got tutorial for usage of heappush and heappop such as heappop(pq)  
    pq = [] 
    h_value = get_manhattan_distance(state)
    heapq.heappush(pq, (h_value, tuple(state), (0, h_value, None)))  # (f, state, (g, h, parent))
    visited_st = {}
    max_length = 0

    while pq:
        curr_f, current_state, (g_curr, h_curr, parent)= heapq.heappop(pq)  #popping lowest 

        if current_state in visited_st:
            continue  #cont if alr visited

        visited_st[current_state] = (parent, g_curr, h_curr) #for marking as visited

        if current_state == tuple(goal_state):
            break  #found goal 

        for succ in get_succ(list(current_state)):
            succ_tuple = tuple(succ)

            if succ_tuple not in visited_st:
                succ_h = get_manhattan_distance(succ) #calc heuristic value
                succ_g = g_curr + 1 #inc cost to reach succ
                succ_f = succ_g + succ_h #tot cost
                heapq.heappush(pq, (succ_f,succ_tuple, (succ_g,succ_h, current_state))) #add succ to pq
        max_length = max(max_length, len(pq)) #for updating max queue length if need be

    curr = tuple(goal_state)
    state_info_list = []

    while curr != tuple(state):
        parent, g, h = visited_st[curr]
        state_info_list.append((list(curr),h,g))
        curr = parent

    state_info_list.append((state, get_manhattan_distance(state), 0))  #adding initia state
    state_info_list.reverse() #so that it shows starting from initial state and towards final state,
    

    # This is a format helper，which is only designed for format purpose.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

