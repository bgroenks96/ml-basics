"""Simple implementation of Tic Tac Toe using approximate Q-Learning."""

import argparse
import copy
import numpy as np
import itertools as it

parser = argparse.ArgumentParser(
    description="Tic Tac Toe with approximate Q-Learning")
parser.add_argument("-b", "--board-size", default=3, type=int,
                    help="Size of the tic tac toe game board (NxN)")
parser.add_argument("-n", "--iterations", default=10, type=int,
                    help="Number of training iterations (games) to run.")
parser.add_argument("-e", "--epsilon", default=0.2, type=float,
                    help="Probability of random behavior while training. \
                    Affects how often the agent will behave sub-optimally \
                    to help 'explore' the state space.")
parser.add_argument("-a", "--alpha", default=0.5, type=float,
                    help="Learning rate. Controls how quickly the agent \
                    updates its weight values.")
parser.add_argument("--alpha-decay", default=0.0001, type=float,
                    help="Learning rate decay. Decreases alpha by the given \
                    amount on each training iteration.")
parser.add_argument("--epsilon-decay", default=0.001, type=float,
                    help="Epsilon decay. Decreases epsilon by the given amount \
                    on each training iteration.")
parser.add_argument("--debug", action="store_true",
                    help="Enables debug mode (fixed random seed)")
args = parser.parse_args()

if args.debug:
    np.random.seed(1)

BOARD_SIZE = args.board_size
# "Spans" in this context are each row, column, and diagonal that "span" the
# entire game board (and constitute goals for both players). It does not have
# any direct correspondence with the meaning of "span" in linear algebra.
BOARD_SPANS = []
board_range = range(BOARD_SIZE)
for i in board_range:
    # Index arrays for row and column i
    r_ind = [[i] * BOARD_SIZE, board_range]
    c_ind = [board_range, [i] * BOARD_SIZE]
    BOARD_SPANS.append(r_ind)
    BOARD_SPANS.append(c_ind)
# Index arrays for both right-left and left-right diagonals
d1_ind = [board_range, board_range]
d2_ind = [np.flip(board_range, 0), board_range]
BOARD_SPANS.append(d1_ind)
BOARD_SPANS.append(d2_ind)

def main():
    # Define player values and create initial board
    p1_x = 1
    p2_o = 2
    initial_state = BoardState()

    # Define features for learning agent
    features = _setup_features(p1_x, p2_o)
    rl_agent = LearningAgent(p1_x, features)
    rand_agent = RandomAgent(p2_o)
    env = environment(rl_agent, rand_agent)
    rl_agent.train(env, initial_state, args.iterations,
                   alpha=args.alpha,
                   epsilon=args.epsilon,
                   epsilon_decay=args.epsilon_decay,
                   alpha_decay=args.alpha_decay,
                   debug_callback=_print_debug)
    print("Training complete. Time for you to play!")
    env = environment(rl_agent, HumanAgent(p2_o))
    rl_agent.train(env, initial_state, 100,
                   alpha=0.5,
                   epsilon=0.0,
                   epsilon_decay=0.0,
                   alpha_decay=args.alpha_decay,
                   debug_callback=_print_debug)


def environment(player, opponent):
    """Return an environment function for the given player and opponent.

    player - an Agent instance for the acting player
    opponent - an Agent instance for the opposing player
    """
    def env(s, a):
        s_n = s.update(a, player.id)
        a_o = opponent.act(s_n)
        if a_o is not None:
            s_r = s_n.update(a_o, opponent.id)
        else:
            s_r = s_n
        r, status = calculate_reward(s_r, player.id)
        return (s_r, r, status)
    return env


def calculate_reward(state, v):
    """Calculate reward for player v based on current state."""
    status = state.get_status()
    r = 0.0
    if status == 0:
        r = -1.0
    elif status == v:
        r = 2.0
    else:
        r = -2.0
    return (r, status)


class BoardState:
    def __init__(self):
        """Create a new BoardState."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))

    def __getitem__(self, xy):
        """Return the value at coordinate xy, i.e. (x,y)."""
        return self.board[xy]

    def update(self, xy, v):
        """Return a new BoardState with coordinate xy, i.e. (x,y), set to v."""
        assert xy[0] >= 0 and xy[0] < BOARD_SIZE
        assert xy[1] >= 0 and xy[1] < BOARD_SIZE
        new_state = BoardState()
        new_state.board = copy.deepcopy(self.board)
        new_state.board[xy] = v
        return new_state

    def count(self, v):
        """Return the number of values on the board matching v."""
        return len(self.board[self.board == v])

    def possible_actions(self):
        """Return an array of possible actions (coordinates) for play."""
        if self.get_status() >= 0:
            return []
        return zip(*np.where(self.board == 0))

    def get_status(self):
        """Return the endgame status of the current state.

        Value will be one of:
        -1 if the current state is not final and additional moves can be made.
        0  if the current state is final and the outcome is a draw.
        x  where x is the id value of the winning player.
        """
        for fs in [self[s] for s in BOARD_SPANS if np.all(self[s] != 0)]:
            # If the span vector is uniform, return the value.
            if np.all(fs == fs[0]):
                return fs[0]
        return 0 if np.all(self.board != 0) else -1

    def dump(self):
        def tostr(x):
            if x == 1.0:
                return 'X'
            elif x == 2.0:
                return 'O'
            else:
                return " "
        for row in self.board:
            print(' '.join([tostr(c) for c in row]))


class LearningAgent:
    def __init__(self, id, features, gamma=0.9):
        self.id = id
        self.features = features
        self.gamma = gamma
        # Initialize weights to length of features + 1 for bias term
        self.weights = np.ones(len(features) + 1)

    def act(self, state):
        actions = state.possible_actions()
        if len(actions) == 0:
            return None
        q_vals = [self._qfunc(state, a) for a in actions]
        a, _ = self._act(state, q_vals, actions, 0.0)
        return a

    def train(self, env, initial_state, itr_count,
              alpha=0.2, epsilon=0.5,
              epsilon_decay=0.01, alpha_decay=0.001, debug_callback=None):
        def q_func(f_vals):
            return np.dot(self.weights, f_vals)
        n = 0
        expected_reward = 0.0
        sum_rewards = 0.0
        while n < itr_count:
            n += 1
            s = initial_state
            actions = s.possible_actions()
            while len(actions) > 0:
                f_vals = np.array([self._feature_values(s, a) for a in actions])
                q_vals = np.array([q_func(f_v) for f_v in f_vals])
                action, ind = self._act(s, q_vals, actions, epsilon)
                s, r, status = env(s, action)
                actions = self._update(s, r, q_vals[ind], f_vals[ind], alpha)
            sum_rewards += r
            expected_reward = sum_rewards / n
            if debug_callback is not None:
                debug_callback(n, s, self.weights, expected_reward, alpha, epsilon)
            # Update values for next iteration
            alpha = max(0.0, alpha - alpha_decay)
            epsilon = max(0.0, epsilon - epsilon_decay)

    def _qfunc(self, state, action):
        return np.dot(self.weights, self._feature_values(state, action))

    def _feature_values(self, state, action):
        # Compute values for features given state and action
        f_vals = [f(state, action) for f in self.features]
        # Append 1 to f_vals to represent bias term
        f_vals = np.append(f_vals, 1)
        return f_vals

    def _update(self, s, r, q_p, f_vals, alpha):
        assert len(f_vals) == len(self.weights)
        actions = s.possible_actions()
        q_vs = np.array([self._qfunc(s, a) for a in actions])
        q_max = np.max(q_vs) if len(q_vs) > 0 else 0
        diff = alpha * (r + self.gamma * q_max - q_p)
        self.weights += diff * f_vals
        return actions

    def _act(self, state, q_vals, actions, epsilon):
        assert len(q_vals) == len(actions)
        if np.random.choice([True, False], p=[1 - epsilon, epsilon]):
            ind = np.argmax(q_vals)
        else:
            ind = np.random.choice(xrange(len(actions)))
        return (actions[ind], ind)


class RandomAgent:
    def __init__(self, id):
        self.id = id

    def act(self, state):
        actions = state.possible_actions()
        if len(actions) == 0:
            return None
        return actions[np.random.choice(xrange(len(actions)))]


class HumanAgent:
    def __init__(self, id):
        self.id = id

    def act(self, state):
        actions = state.possible_actions()
        if len(actions) == 0:
            return None
        state.dump()
        valid = False
        while not valid:
            val = input('Specify coordinate for your move x,y: ')
            valid = isinstance(val, tuple) \
                    and all([isinstance(x, int) for x in val]) \
                    and val in actions
            if not valid:
                print('Invalid value. Try again.')
        return val


def feature_count_moves(v):
    """Return a feature function for counting moves.

    Counts number of moves on the board for v.
    """
    max_moves = BOARD_SIZE**2 / 2.0
    return lambda s, a: _normalize(s.update(a, v).count(v), 0, max_moves)


def feature_score_span(v):
    """Return a feature function for scoring board spans for v.

    The score is increased for each index of the vector in the board that has a
    value matching v and decreased for each non-zero value that does
    not match v.
    """
    def func(s, a):
        s_n = s.update(a, v)
        raw_score = 0
        for i in BOARD_SPANS:
            t = s_n[i]
            t_score = 0
            for x in t:
                if x == v:
                    t_score += 1
                elif x != 0:
                    t_score -= 1
            raw_score += t_score ** 3
        max_score = BOARD_SIZE * BOARD_SIZE * len(BOARD_SPANS)
        return _normalize(raw_score, -max_score, max_score)
    return func

# An array of delta coordinates (0,-1), (0,1), (-1, 0), etc. that can be used
# to get the neighbors of an index. Note that we drop the first element (0,0)
# so that we don't count the point itself as a neighbor.
neighbor_deltas = np.array(list(it.product([0,-1,1], repeat=2)))[1:]


def feature_score_nearest_neighbors(v):
    """Return a feature function for scoring nearest neighbors.

    Evaluates the action for a given state by increasing the score for empty
    or "friendly" neighbors and decreasing it for neighbors marked by the
    opponent.
    """
    def func(s, a):
        s_n = s.update(a, v)
        neighbors = [xy for xy in (neighbor_deltas + a)
                     if np.all(xy >= 0) and np.all(xy < BOARD_SIZE)]
        neighbor_indices = zip(*neighbors)
        neighbor_vals = s_n[neighbor_indices]
        neighbor_scores = [1.0 if x == 0 or x == v else 0.0
                           for x in neighbor_vals]
        raw_score = np.sum(neighbor_scores)
        max_score = 2**BOARD_SIZE
        return _normalize(raw_score, 0, max_score)
    return func


def _normalize(v, min, max):
    """Normalize v with range min-max to range of 0.0-1.0."""
    return float(v - min) / float(max - min)


def _setup_features(p_id, op_id):
    features = []
    # We add each feature once per player. This allows the learning agent
    # to "see" both its own standing and its opponent's standing relative to
    # each feature.
    features.append(feature_score_nearest_neighbors(p_id))
    features.append(feature_score_nearest_neighbors(op_id))
    features.append(feature_score_span(p_id))
    features.append(feature_score_span(op_id))
    # Span score feature
    # for s in BOARD_SPANS:
    #    features.append(feature_score_span(p_id, s))
    #    features.append(feature_score_span(op_id, s))
    return features

def _print_debug(n, s, weights, expected_reward, alpha, epsilon):
    print("Iteration: {0} (alpha={1}, epsilon={2})".format(n, alpha, epsilon))
    s.dump()
    print("Weights: {0}".format(weights))
    print("Expected reward: {0}".format(expected_reward))
    print("----------------")

if __name__ == "__main__":
    main()
