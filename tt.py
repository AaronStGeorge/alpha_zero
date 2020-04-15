import math
import numpy
from typing import List
import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AlphaZeroConfig(object):

    def __init__(self):
        # Self-Play ==
        self.num_actors = 5000

        self.num_sampling_moves = 30
        self.max_moves = 42  # 512 for chess and shogi, 722 for Go.
        self.num_simulations = 20 # 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training ==
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(50)
        self.batch_size = 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }


class Node(object):

    def __init__(self, prior: float):  # prior = how good the network thought it would be
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):

    def __init__(self, history=None):
        # Connect 4 specific ===
        self._num_rows = 6
        self._num_cols = 7

        self._winner = None

        # Masks to "convolve" over board and detect a winner
        self._win_masks = []
        # Horizontal wins
        for i in range(4):
            mask = np.zeros((4, 4), dtype=np.bool)
            mask[i, :] = True
            self._win_masks.append(mask)
        # Vertical wins
        for j in range(4):
            mask = np.zeros((4, 4), dtype=np.bool)
            mask[:, j] = True
            self._win_masks.append(mask)
        # Diagonal wins
        down = np.zeros((4, 4), dtype=np.bool)
        for i, j in zip(range(4), range(4)):
            down[i, j] = True
        self._win_masks.append(down)
        up = np.zeros((4, 4), dtype=np.bool)
        for i, j in zip(reversed(range(4)), range(4)):
            up[i, j] = True
        self._win_masks.append(up)

        # All games will have these ===
        self.history = history or []
        self.child_visits = []
        self.num_actions = self._num_cols  # 7 for connect 4, 512 for chess/shogi, and 722 for Go.

    def terminal(self):
        """
        returns bool if the game is finished or not
        """
        if self._winner is not None or len(self.history) == 42:
            return True

        image = self.make_image(len(self.history))
        # check for wins from the bottom of the board up. Wins are more likely to appear there.
        for i in reversed(range(self._num_rows - 3)):
            for j in range(self._num_cols - 3):
                for mask in self._win_masks:
                    for player in range(2):
                        test = image[player, i:i + 4, j:j + 4][mask]
                        if np.alltrue(test == 1):
                            self._winner = player
                            return True

        return False

    def terminal_value(self, to_play):
        """
        The result of the game from the player that's going to_play? If player 1
        won then and to_play is 1 then return 1 if to_play is 2 then return -1?
        """
        if self._winner is None and len(self.history) == 42:
            return 0
        return to_play == self._winner

    def legal_actions(self):
        image = self.make_image(len(self.history))
        return [j for j in range(self._num_cols) if image[0, 0, j] == 0 and image[1, 0, j] == 0]

    def clone(self):
        return Game(list(self.history))

    def apply(self, action: int):
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in iter(root.children.values()))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        """
        returns what the game looked like at state_index i
        """
        player_0 = np.zeros((self._num_rows, self._num_cols), dtype=numpy.float)
        player_1 = np.zeros((self._num_rows, self._num_cols), dtype=numpy.float)
        for move_i, move in enumerate(self.history[:state_index+1]):
            for row in reversed(range(self._num_rows)):
                if player_0[row, move] == 0 and player_1[row, move] == 0:
                    if move_i % 2 == 0:
                        player_0[row, move] = 1
                    if move_i % 2 == 1:
                        player_1[row, move] = 1
                    break

        to_play = (state_index + 1) % 2 * np.ones((self._num_rows, self._num_cols), dtype=numpy.float)

        return np.array([player_0, player_1, to_play], dtype=numpy.float)

    def make_target(self, state_index: int):
        """
        returns the nural network target i.e. what the NN should be gessing given the image
        """
        return (self.terminal_value(state_index % 2),  # state_index % 2 will always be who's playing
                self.child_visits[state_index])

    def to_play(self):
        """
        Return the player that is about to play
        """
        return len(self.history) % 2

    def __str__(self):
        board_state = self.make_image(len(self.history))

        out = ""
        for i in range(self._num_rows):
            out += f"{i}|"
            for j in range(self._num_cols):
                if board_state[0, i, j] == 1:
                    out += " ○ "
                elif board_state[1, i, j] == 1:
                    out += " ● "
                else:
                    out += "   "
            out += "|\n"

        out += "  "
        for j in range(self._num_cols):
            out += f" \u0305{j} "
        return out


class ReplayBuffer(object):

    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = numpy.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]

        image = np.array([g.make_image(i) for (g, i) in game_pos], dtype=np.float)
        image = torch.from_numpy(image)
        image = image.to(torch.float)

        policy_target = np.array([g.make_target(i)[1] for (g, i) in game_pos])
        policy_target = torch.from_numpy(policy_target)
        policy_target = policy_target.to(torch.float)

        value_target = np.array([g.make_target(i)[0] for (g, i) in game_pos])
        value_target = torch.from_numpy(value_target)
        value_target = value_target.to(torch.float)

        batch_data = TensorDataset(image, policy_target, value_target)
        return torch.utils.data.DataLoader(dataset=batch_data,
                                           batch_size=50,
                                           shuffle=True)


class Network(object):

    def inference(self, image):
        return -1, {0: 1 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 1 / 7, 5: 1 / 7, 6: 1 / 7}  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return []

class Net(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(Net, self).__init__()

        """
        Shared
        """
        self.shared_conv = nn.Sequential(
            # Conv Layer block 1
            #  input shape:
            #    3, 6, 7
            #  output shape:
            #    32, 6, 7
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


        """
        Value Head
        """
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 6 * 7, 120),
            nn.RReLU(inplace=True),
            nn.Linear(120, 1),
        )

        """
        Policy Head
        """
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(32 * 6 * 7, 120),
            nn.RReLU(inplace=True),
            nn.Linear(120, 7),
        )

    def inference(self, image):
        image = torch.from_numpy(image)
        image = image.to(torch.float)
        # TODO: why the fuck do we have to squeeze and unsqueeze everything all the damn time
        image = image.unsqueeze(0)
        # TODO: make p and v ordering consistent
        p, v = self.forward(image)
        return float(v.squeeze().detach()), p.squeeze().detach().numpy()

    def forward(self, x):
        """Perform forward."""

        # shared conv
        sc = self.shared_conv(x)

        # policy head ==
        p = self.policy_conv(sc)

        # flatten
        p = p.view(p.size(0), -1)

        # fc layer
        p = self.policy_fc(p)

        # value head ==
        v = self.value_conv(sc)

        # flatten
        v = v.view(v.size(0), -1)

        # fc layer
        v = self.value_fc(v)

        return p, v


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig, network: Net):
    replay_buffer = ReplayBuffer(config)

    cycles = 50
    for i in range(cycles):
        print(f"self play {i} of {cycles}")
        network.eval()
        run_selfplay(config, network, replay_buffer)
        print(f"train network {i} of {cycles}")
        network.train()
        train_network(config, replay_buffer)

    return network


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, network: Net,
                 replay_buffer: ReplayBuffer):
    for i in range(config.window_size):  # TODO: make better
        if i % 10 == 0:
            print(f"game {i} of {config.window_size}")
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Net):
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Net):
    root = Node(0)
    # Populate child nodes AKA the states that the actions available at this
    # states would take you too
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for i in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            # Here we take one step down our search tree towards a win or loss. Note
            # that we are resetting the node variable here to be the state that our
            # game picked given the action we took.
            #
            # On the first run all child nodes will not be expanded, so we'll only
            # take one step before backpropatagating back up the tree. This is a form
            # of "bootstrapping"
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in iter(root.children.items())]
    # TODO: does this even make sense for connect 4?
    # if len(game.history) < config.num_sampling_moves:
    #     _, action = softmax_sample(visit_counts)
    # else:
    _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    """
    Return the child node, i.e. action to take, that UCB likes best
    """
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in iter(node.children.items()))
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Net):
    """
    Populate child nodes with priors and return value both derived from NN
    Child nodes are the states that one could reach by taking the actions
    available from the state that you are in.
    """
    value, policy_logits = network.inference(game.make_image(len(game.history)))

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(iter(policy.values()))
    for action, p in iter(policy.items()):
        # this is just softmax, notice the math.exp 3 lines up
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    """
    Modifies the priors stored in nodes children with dirichlet noise whatever
    that is
    """
    actions = node.children.keys()
    noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


class board_data(Dataset):
    """
    This turns the np tensor into a torch dataset
    """

    def __init__(self, dataset):
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return np.int64(self.X[index].transpose(2, 0, 1)), self.y_p[index], self.y_v[index]


class ConvBlock(nn.Module):
    """
    Here we define the convolution block
    """

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)
        s = self.conv1(s)
        s = self.bn1(s)
        s = nn.functional.relu(s)
        return s


class ResBlock(nn.Module):
    """
    Here we define the ResNet blocks
    """

    def __init__(self, in_planes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = nn.functional.relu(out)

        return out


class DualHead(nn.Module):
    """
    Here we define the two heads of the NN, the policy head and the value head
    """

    def __init__(self):
        super(DualHead, self).__init__()
        """
        Value Head
        """
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 6 * 7, 32)
        self.value_fc2 = nn.Linear(31, 1)

        """
        Policy Head
        """
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_log_softmax = nn.LogSoftmax(dim=1)
        self.policy_fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s):
        """
        Value Head
        """
        v = self.value_conv(s)
        v = self.value_bn(v)
        v = nn.functional.relu(v)
        v = v.view(-1, 3 * 6 * 7)
        v = self.value_fc1(v)
        v = nn.functional.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)

        """
        Policy Head
        """
        p = self.policy_conv(s)
        p = self.policy_bn(p)
        p = nn.functional.relu(p)
        p = p.view(-1, 6 * 7 * 32)
        p = self.policy_fc(p)
        p = self.policy_log_softmax(p).exp()

        return p, v


class ConnectNet(nn.Module):
    """
    Here we bring the ResNet blocks and the dual headed network together
    """

    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(10):
            setattr(self, "res_%i" % block, ResBlock())
        self.out_block = DualHead()

    def forward(self, s):
        s = self.conv(s)
        for block in range(10):
            s = getattr(self, "res_i%" % block)(s)  # this looks a little funny
        s = self.out_block(s)

        return s


class AlphaLoss(nn.Module):
    """
    Here we define the loss function
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, z, value, pi, policy, theta):
        value_error = (z - value) ** 2
        policy_error = nn.functional.binary_cross_entropy_with_logits(policy, pi)
        # we take care of L2 regularization in the optimizer

        return value_error - policy_error


def train_network(config: AlphaZeroConfig, replay_buffer: ReplayBuffer):
    # optimizer = torch.optim.SGD(network.parameters(), momentum=0.9, weight_decay=10e-4)
    optimizer = torch.optim.Adam(network.parameters())

    for i in range(1): #(config.training_steps):
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch)

    # optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
    #                                        config.momentum)
    # for i in range(config.training_steps):
    #     if i % config.checkpoint_interval == 0:
    #         storage.save_network(i, network)
    #     batch = replay_buffer.sample_batch()
    #     update_weights(optimizer, network, batch, config.weight_decay)
    # storage.save_network(config.training_steps, network)


def update_weights(optimizer, network, batch):
    # Loop over each subset of data
    for image, policy_target, value_target in batch:
        # Zero out the optimizer's gradient buffer
        optimizer.zero_grad()

        # Make a prediction based on the model
        policy, value = network(image)

        # Compute the loss
        # .sum() is some black magic https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152/2
        loss = (nn.functional.binary_cross_entropy_with_logits(policy, policy_target) + (value_target - value) ** 2).sum()

        # Use backpropagation to compute the derivative of the loss with respect to the parameters
        loss.backward()

        # Use the derivative information to update the parameters
        optimizer.step()




    # loss = 0
    # for image, (target_value, target_policy) in batch[:1]:
    #     thing = torch.from_numpy(image)
    #     thing = thing.to(torch.float)
    #     thing = thing.unsqueeze(0)
    #     policy = network(thing)
        # loss += AlphaLoss(target_value, value, target_policy, policy)

    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    #     value, policy_logits = network.inference(image)
    #     loss += (
    #             tf.losses.mean_squared_error(value, target_value) +
    #             tf.nn.softmax_cross_entropy_with_logits(
    #                 logits=policy_logits, labels=target_policy))
    #
    # for weights in network.get_weights():
    #     loss += weight_decay * tf.nn.l2_loss(weights)
    #
    # optimizer.minimize(loss)



######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
    return 0, 0


def make_uniform_network():
    network = Net()
    network.float()
    return network


def interactive_game(config: AlphaZeroConfig, network: Network):
    game = Game()
    print(game)
    while not game.terminal():
        while True:
            print("choose move please: ", end='')
            human_action = input()
            try:
                if int(human_action) not in game.legal_actions():
                    print("illegal action")
                else:
                    break
            except ValueError:
                print("illegal action")
        game.apply(int(human_action))
        # print(game)
        ai_action, _ = run_mcts(config, game, network)
        print(f"ai chooses {ai_action}")
        game.apply(ai_action)
        print(game)
    win_string = {-1: "lost", 1: "won", 0: "tied"}
    print(f"you {win_string[game.terminal_value(0)]}")
    print(game)


if __name__ == "__main__":
    network = make_uniform_network()
    config = AlphaZeroConfig()
    # interactive_game(config, network)
    alphazero(config, network)
    interactive_game(config, network)
