'''
This game is the Chopstick game, a turn-based game where each player start with 1 finger at each hand.
At each turn, the player can choose to either attack or transfer.
For an attack, the player chooses one of his hands as the weapon and one of his opponent's hands as the target.
Both the weapon and the target must have positive number of fingers.
As a result, the target increases its number of fingers by the number of fingers on the weapon.
If the number of fingers of the target is at least 5, the number of fingers of the target becomes zero.
For a transfer, the player chooses one of his hands as the starting hand and his other hand as the destination.
If the destination has zero finger, the player cannot transfer all fingers from the starting hand.
Number of fingers transferred must be at least one.
In a legal transfer, the destination after the transfer cannot hold 5 or more fingers.
A player loses when both of his hands have zero fingers.

In this code sample, I trained the AI using reinforcement learning.
The game is represented by the ChopSticks class, which consists of the fields of state, turn and move_count
The state is represented by an array of arrays [[p0h0, p0h1], [p1h0, p1h1]],
where each element represents the number of fingers on a person's hand.
The AI is represented by the ChopSticksAI class, which consists of
alpha, the learning coefficient,
epsilon, the probability that the AI takes a random move during training,
q, a dictionary storing the insights of the AI on the game, each key is a pair of game state and action
ChopSticksAI also supports the method combine(another_AI), which combines its own insights with another AI
train(n) method is meant to train the agents with n games.
The minimum recommended number of trainings for the AI to be absolutely smart is 20000.
finally, play() method is used to train the agents and test.
'''

import random
from numpy import e


start = 1
max_fingers = 5
max_move_count = 100
n_hands = 2


class ChopSticks():
    def __init__(self):
        self.state = [[start for i in range(n_hands)] for j in range(2)]
        self.turn = 0 # keep track of whose turn it is
        self.move_count = 0 # limit the number of moves in a game to avoid an eternal loop
    
    def print_game_state(self):
        ''' this function should only be called during testing, where turn 1 represents human, turn 0 represents AI '''
        print()
        for i in range(n_hands):
            print(f'your hand {i}: {self.state[1][i]}')
        for i in range(n_hands):
            print(f'AI hand {i}: {self.state[0][i]}')
    
    def human_turn(self):
        ''' this function should only be called during testing, return true if turn is 1 '''
        return self.turn == 1
    
    def set_turn(self, go_first):
        ''' only used during playtesting, to start the game '''
        if go_first == 'y':
            self.turn = 1
        else:
            self.turn = 0
    
    def actions(self):
        ''' output is a list
        each element has the first sub-element denoting whether it is an attack or transfer
        for attack, each element has 2 other sub-elements, the starting hand and the ending hand
        for transfer, each element has 2 other sub-elements, the starting hand and number of fingers transferred '''
        possible_moves = []
        # attack
        for my_hand_index in range(n_hands):
            if self.state[self.turn][my_hand_index] > 0:
                for opp_hand_index in range(n_hands):
                    if self.state[(self.turn + 1) % 2][opp_hand_index] > 0:
                        possible_moves.append(('attack', my_hand_index, opp_hand_index))
        # transfer
        for hand_index in range(n_hands):
            if self.state[self.turn][hand_index] > 0:
                for finger_count in range(1, self.state[self.turn][hand_index] + 1):
                    if finger_count == self.state[self.turn][hand_index] and self.state[self.turn][(hand_index + 1) % 2] == 0:
                        break
                    if finger_count + self.state[self.turn][(hand_index + 1) % 2] < max_fingers:
                        possible_moves.append(('transfer', hand_index, finger_count))
                    else:
                        break
        return possible_moves
    
    def make_move(self, move):
        ''' reset the number of fingers where possible
        rotate the turn and add 1 to move_count '''
        if move[0] == 'attack':
            finger_count = self.state[self.turn][move[1]]
            self.state[(self.turn + 1) % 2][move[2]] += finger_count
            if self.state[(self.turn + 1) % 2][move[2]] >= max_fingers:
                self.state[(self.turn + 1) % 2][move[2]] = 0
        
        if move[0] == 'transfer':
            finger_count = move[2]
            self.state[self.turn][(move[1] + 1) % 2] += finger_count
            self.state[self.turn][move[1]] -= finger_count
        
        self.turn = (self.turn + 1) % 2
        self.move_count += 1
    
    def terminal(self):
        ''' either one player loses, or the game is at a stalemate '''
        if self.winner() != None:
            return True
        
        if self.move_count > max_move_count:
            return True
        
        return False
    
    def winner(self):
        ''' determine the winner, assuming that terminal() returns True
        return None if there is no winner '''
        for player in range(2):
            dead = True
            for hand_index in range(n_hands):
                if self.state[player][hand_index] != 0:
                    dead = False
                    break
            if dead:
                return (player + 1) % 2
        
        return None


class ChopSticksAI():
    def __init__(self):
        self.alpha_min = 0.3
        self.alpha_max = 0.8
        self.epsilon_max = 0.3
        self.slope = 0.0005
        self.game_count = 0
        self.q = dict()
        self.moves = []
    
    def alpha(self, closeness):
        ''' find the learning coefficient with closeness to the end of the game, 1 being the very end and 0 being the very start'''
        return self.alpha_min + (self.alpha_max - self.alpha_min) * closeness
    
    def epsilon(self):
        return self.epsilon_max # self.epsilon_max * e ** (-self.slope * self.game_count)
    
    def turn_tuple(self, state):
        ''' turn a state into tuple for dictionary purposes '''
        rstate = []
        for i in range(2):
            rstate.append(tuple(state[i]))
        return tuple(rstate)
    
    def get_q_value(self, state, action):
        ''' get q value from the dictionary, the state and action pair may not be in the dictionary yet '''
        try:
            return self.q[self.turn_tuple(state), action]
        except:
            return 0
    
    def update_q_value(self, state, action, new_q, closeness):
        ''' update the dictionary when a reward is given to a particular state and action pair '''
        self.q[self.turn_tuple(state), action] = self.get_q_value(state, action) + self.alpha(closeness) * (new_q - self.get_q_value(state, action))
    
    def update_q_values(self, game_result):
        n_moves = len(self.moves)
        reward = 0
        if game_result == "win":
            reward = 1
        elif game_result == "lose":
            reward = -1
        elif game_result == "draw":
            reward = 0
        for i in range(n_moves):
            self.update_q_value(self.moves[i][0], self.moves[i][1], reward, i/(n_moves - 1))
        self.refresh()
    
    def choose_action(self, game, epsilon=True):
        ''' turn off epsilon in real matches '''
        possible_moves = game.actions()
        def best_action():
            ''' determine the best action in the state of the game '''
            q_value = -2 # starting q-value, after an action is chosen, the q-value follows the q value of the state-action pair
            for action in possible_moves:
                if self.get_q_value(game.state, action) > q_value:
                    chosen_action = action
                    q_value = self.get_q_value(game.state, action)
            return chosen_action
        
        if epsilon:     # during training
            p = random.random()
            if p < self.epsilon():
                # return random action
                ind = random.randrange(0, len(possible_moves))
                self.moves.append((self.turn_tuple(game.state), possible_moves[ind]))
                return possible_moves[ind]
            else:
                action = best_action()
                self.moves.append((self.turn_tuple(game.state), action))
                return action
        else:
            return best_action()
    
    def refresh(self):
        self.moves = []
        self.game_count += 1
    
    def combine(self, another_AI):
        ''' combine the insights of the AIs '''
        difference = []
        for state, action in list(another_AI.q.keys()):
            if ((state[1], state[0]), action) in self.q.keys():
                self.q[(state[1], state[0]), action] = (self.q[(state[1], state[0]), action] + another_AI.q[state, action])/2
                difference.append(abs(self.q[(state[1], state[0]), action] - another_AI.q[state, action]))
            else:
                self.q[(state[1], state[0]), action] = another_AI.q[state, action]
        
        avg_difference = 10
        max_difference = 10
        try:
            avg_difference = sum(difference)/len(difference)
            max_difference = max(difference)
        except:
            pass
        return (avg_difference, max_difference)


AI = ChopSticksAI()
AI2 = ChopSticksAI()
AIs = [AI, AI2]

def train(n):
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f'training game {i + 1}')
        
        game = ChopSticks()
        while not game.terminal():
            action = AIs[game.turn].choose_action(game)
            game.make_move(action)
            
        # reward the AIs accordingly
        winner = game.winner()
        if winner == 0 or winner == 1:
            loser = (winner + 1) % 2
            AIs[winner].update_q_values("win")
            AIs[loser].update_q_values("lose")
        else:
            for player in range(2):
                AIs[player].update_q_values('draw')


def play():
    ''' interact with the machine '''
    n_trainings = int(input('number of trainings? '))
    train(n_trainings)
    print()
    print('training finished')
    print()
    print("combining insights ...")
    avg_difference, max_difference = AI.combine(AI2)
    print(f'avg difference in insights: {avg_difference}')
    print(f'max difference in insights: {max_difference}')
    
    game = ChopSticks()
    response = input('play first (y/n)? ')
    game.set_turn(response)
    
    while not game.terminal():
        print()
        game.print_game_state()
        
        if game.human_turn():  # human turn
            response = input('attack or transfer (a/t)? ')
            if response == 'a':
                attack = int(input('choose your hand to attack: '))
                receiver = int(input("choose the opponent's hand: "))
                game.make_move(('attack', attack, receiver))
            if response == 't':
                transferer = int(input('choose your starting hand: '))
                n_fingers = int(input('choose number of fingers to be transferred: '))
                game.make_move(('transfer', transferer, n_fingers))
        else:   # AI's turn
            action = AI.choose_action(game, epsilon = False)
            for move in game.actions():
                print(move)
                print(AI.get_q_value(game.state, move))
            game.make_move(action)
            if action[0] == 'attack':
                print(f'AI used hand {action[1]} to attack your hand {action[2]}')
            else:
                print(f'AI transferred {action[2]} fingers from hand {action[1]}')
    
    print()
    if game.winner() == 1:
        print('winner is human!')
    elif game.winner() == 0:
        print('winner is AI')
    else:
        print('the game is a draw')


if __name__ == '__main__':
    play()