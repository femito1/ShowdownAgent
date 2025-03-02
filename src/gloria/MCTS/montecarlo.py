import numpy as np
from poke_env.environment import Battle, observation
from gloria.embedding.get_embeddings import MOVES, POKEMONS, GlorIA
from gloria.MCTS.simulator import Simulator
# MISSING FUNCTION AND IMPORTS:
n_options = 5 #len(POKEMONS) + len(MOVES)

def model(state): 
    '''
    probability of the model choosing any action (everything is masked except that action)
    in a state
    '''
    res = [0.1]*n_options

    return res

def can_moves(state):
    '''
    calculate all the available actions from a given state
    '''

    if np.random.randint(0, 100) == 0:
        res =  np.zeros(n_options)
    else:
        res = np.ndarray(0,2, (n_options))
    
    return res

def critic_head(state):
    '''
    evaluates how good a state is
    '''
    res = np.count_nonzero([1,1,0,0,0])/n_options
    return res

# ASSUMPTION: available actions should be a np array of 1s and 0s masking all possible moves 
# and pokemon to switch out to:
# [pokemon switch, moves]
# since POKEMON is 296 long and MOVES is 188, available actions are 484

# HYPERPARAMETERS
beta = 0.5 # between 0-1 
alpha = 0.1 # between 0-1 
#


class Node():
    def __init__(self, s: np.array):
        '''
        Nodes should initialize all their values and of their corresponding
        actions and then be updated at every rollout
        '''
        self.state: Battle = s # battle state
        self.M = 0 # times this node was visited during rollouts
        available_actions = can_moves(s)
        self.isterminal = not np.count_nonzero(available_actions) > 0 # is leaf or terminal?
        self.actions: np.array = self._encode_actions(available_actions) # actions available from state s
        self.encoding = GlorIA.embed_battle(s).flatten()

    def update(self, action, v):
        '''
        update value of the node during backpropagation
        '''
        action.U = action.P ** beta * ((self.M)**(1/2))/(action.N + 1)
        action.Q = (action.N * action.Q + v) / (action.N + 1)
        action.N = (action.N + 1)
        self.M = self.M + 1
    
    def _encode_actions(self, available_actions: np.array):
        prob_for_actions: list[float] = model(self.state)
        for i in range(len(available_actions)):
            if available_actions[i] == 1:
                available_actions[i] = action(i, prob_for_actions[i]) 

        return np.array(available_actions)
    
    def __eq__(node1, node2):
        res = np.array_equal(node1.encoding, node2.encoding)
        return res

    def __hash__(self):

        obs = self.encoding

        '''obs = np.array(map(lambda x: x.events, self.state.observations.values())).flatten()''' # this encodes the actual perfect state

        to_encode = tuple(obs)

        return hash(to_encode)
        """
        tuple(self.state.available_moves), 
        tuple(self.state.available_switches),
        tuple(self.state.force_switch),
        tuple(self.state.fields),
        tuple(self.state.finished),
        tuple(self.state.won),
        tuple(map(str, self.state.weather.keys())), 
        tuple(self.state.weather.values()),
        tuple(self.state.trapped),
        tuple(self.state.maybe_trapped),


        tuple(map(str, self.state.opponent_side_conditions.keys())), # to get the names of side conditions #are they all initialized at 0? i don't think so
        tuple(self.state.opponent_side_conditions.values()),
        tuple(self.state.opponent_team.values()),
        tuple(map(str,self.state.opponent_active_pokemon)),
        
        tuple(map(str, self.state.side_conditions.keys())), 
        tuple(self.state.side_conditions.values()),
        tuple(self.state.team.values),
        tuple(map(str, self.state.active_pokemon)),"""
         
class action:
    def __init__(self, available_action: int, P: float):
        # available_action is just the index of the only 1
        self.a = available_action

        self.Q = 0 # reward for action
        # probability of taking this action, 
        # available action is all the actions masked minus one
        self.P = P # probability of the model taking this action
        self.N = 0 # times action was taken
        self.U = 0 # upper confidence bound for this action

    def __hash__(self):
        return hash(self.a, self.Q,  self.P, self.N, self.U)

class MCTS():
    def __init__(self, state: Battle):
        self.simulator = Simulator()
        self.root = Node(state)
        self.children: dict[Node, list[Node]] = {self.root: None}


    def rollout(self):
        path = self._choose(self.root) # find path to a leaf/terminal node
        leaf = path[-1] 
        self._backpropagate(path, critic_head(leaf.encoding))

    def act(self):
        '''
        once rollouts are done return the best action to take, and wait for the next
        battlestate
        '''
        action = np.argmax(np.array(map(lambda x: x.N, self.root.actions)))
        return self.simulator.decode_action(action.a) # the decoded action to take, DEPENDS ON THE IMPLEMENTATION OF HOW GLORIA WILL TAKE ACTIONS

    def new_root(self, state: Battle):

        old_root = self.root

        new_root_idx = self.root.encoding == np.array(map(lambda x: x.encoding, self.children[self.root]))
        if not np.count_nonzero(new_root_idx): 
            # somehow we did not see this state previously, create new tree
            self.root = Node(state)
        else:
            self.root = self.children[self.root][np.argmax(new_root_idx)]

        self.children.pop(old_root) 
            
    def _backpropagate(self, path: list[Node], v):
        for node, action in path:
            node.update(action, v)

    def _choose(self, node: Node) -> list:
        '''
        find the path from root to leaf/terminal node
        return the list of the path
        '''
        path = list()
        while True:
            expand_action: int = np.argmax(map(lambda x: 0 if x == 0 else x.U, node.actions)) # array is either 0 or action object
            path.append(node, node[expand_action])
            
            next_state: Battle = self.simulator(node.state, self.simulator.sim(expand_action)) # expand action is an index of the move
            new_node = Node(next_state)

            states_found = new_node.encoding == np.array(map(lambda x: x.encoding, self.children[node]))
            if not np.count_nonzero(states_found) or new_node.isterminal: 
                # means that we never visited this state or that it is terminal
                if not self.children[node]:
                    self.children[node] = [new_node]
                else:
                    self.children[node].append(new_node)
                break # we break out of the loop and backpropagate
            node = self.children[node][np.argmin(states_found)] # if we have it in the children we expand it and go on
        return path



    

    