import random
import time
import curses

GRID_SIZE = 10
MOVE_DELAY = 0.1
ACTIONS = ['w','a','s','d']
OPPOSITES = {'w':'s','s':'w','a':'d','d':'a'}

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

REWARD_FOOD = 100
REWARD_CLOSER = 50
REWARD_FARTHER = -50
REWARD_DEATH = -100
PROXIMITY_PENALTY = 30

class SnakeGame:
    def __init__(self):
        self.snake = []
        self.direction = 'd'
        self.done = False
        self.food = ()
        self.reset()

    def reset(self):
        self.snake = [(5,5),(5,4),(5,3)]
        self.direction = 'd'
        self.done = False
        self.spawn_food()

    def spawn_food(self):
        while True:
            self.food = (random.randint(1, GRID_SIZE), random.randint(1, GRID_SIZE))
            if self.food not in self.snake:
                break
    def distance_to_food(self):
        hy,hx = self.snake[0]
        fy,fx = self.food
        return abs(hy-fy) + abs(hx-fx)

    def step(self, action):
        if action == OPPOSITES[self.direction]:
            action = self.direction
        self.direction = action

        hy,hx = self.snake[0]
        if action == 'w': new_head = (hy-1,hx)
        elif action == 's': new_head = (hy+1,hx)
        elif action == 'a': new_head = (hy,hx-1)
        else: new_head = (hy,hx+1)

        old_dist = self.distance_to_food()

        # Collision
        if (new_head[0]==0 or new_head[0]==GRID_SIZE+1 or
            new_head[1]==0 or new_head[1]==GRID_SIZE+1 or
            new_head in self.snake):
            self.done = True
            return REWARD_DEATH

        self.snake.insert(0,new_head)

        reward = 0
        if new_head == self.food:
            self.spawn_food()
            reward += REWARD_FOOD
        else:
            self.snake.pop()
            new_dist = self.distance_to_food()
            reward += REWARD_CLOSER if new_dist < old_dist else REWARD_FARTHER

        # Danger checks (ignore first 3 tail pieces)
        danger_w = 1 if hy-1==0 or (hy-1,hx) in self.snake[3:] else 0
        danger_s = 1 if hy+1==GRID_SIZE+1 or (hy+1,hx) in self.snake[3:] else 0
        danger_a = 1 if hx-1==0 or (hy,hx-1) in self.snake[3:] else 0
        danger_d = 1 if hx+1==GRID_SIZE+1 or (hy,hx+1) in self.snake[3:] else 0

        if action=='w' and danger_w: reward -= PROXIMITY_PENALTY
        if action=='s' and danger_s: reward -= PROXIMITY_PENALTY
        if action=='a' and danger_a: reward -= PROXIMITY_PENALTY
        if action=='d' and danger_d: reward -= PROXIMITY_PENALTY

        return reward

    def render(self, win):
        win.clear()
        win.border()
        for y,x in self.snake:
            win.addch(y,x,'#')
        fy,fx = self.food
        win.addch(fy,fx,'*')
        win.refresh()
class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Small initial weights to avoid early bias
        self.w1 = [[random.uniform(-0.1,0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [[random.uniform(-0.1,0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0 for _ in range(output_size)]

    def relu(self,x): return [max(0,xi) for xi in x]
    def relu_derivative(self,x): return [1 if xi>0 else 0 for xi in x]

    def forward(self,inp):
        self.inp = inp[:]
        self.h = [sum(self.inp[i]*self.w1[i][j] for i in range(self.input_size))+self.b1[j]
                  for j in range(self.hidden_size)]
        self.h_act = self.relu(self.h)
        self.out = [sum(self.h_act[i]*self.w2[i][j] for i in range(self.hidden_size))+self.b2[j]
                    for j in range(self.output_size)]
        return self.out

    def train(self,target):
        out_grad = [self.out[i]-target[i] for i in range(self.output_size)]
        # Update w2 and b2
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.w2[i][j] -= self.lr*out_grad[j]*self.h_act[i]
        for j in range(self.output_size):
            self.b2[j] -= self.lr*out_grad[j]
        # Backprop to hidden
        hidden_grad = [sum(out_grad[j]*self.w2[i][j] for j in range(self.output_size))*d
                       for i,d in enumerate(self.relu_derivative(self.h))]
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w1[i][j] -= self.lr*hidden_grad[j]*self.inp[i]
        for j in range(self.hidden_size):
            self.b1[j] -= self.lr*hidden_grad[j]

class Agent:
    def __init__(self,input_size,hidden_size,output_size):
        self.net = NeuralNet(input_size,hidden_size,output_size)
        self.epsilon = EPSILON_START

    def get_state_vector(self,game):
        hy,hx = game.snake[0]
        fy,fx = game.food
        dy = 0 if fy==hy else (1 if fy>hy else -1)
        dx = 0 if fx==hx else (1 if fx>hx else -1)
        # normalized distances
        dy_dist = (fy-hy)/GRID_SIZE
        dx_dist = (fx-hx)/GRID_SIZE
        danger_w = 1 if hy-1==0 or (hy-1,hx) in game.snake[3:] else 0
        danger_s = 1 if hy+1==GRID_SIZE+1 or (hy+1,hx) in game.snake[3:] else 0
        danger_a = 1 if hx-1==0 or (hy,hx-1) in game.snake[3:] else 0
        danger_d = 1 if hx+1==GRID_SIZE+1 or (hy,hx+1) in game.snake[3:] else 0
        return [dy,dx,dy_dist,dx_dist,danger_w,danger_s,danger_a,danger_d]

    def choose_action(self,state_vec):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = self.net.forward(state_vec)
        max_idx = q_values.index(max(q_values))
        return ACTIONS[max_idx]

    def train_step(self,state_vec,action,reward,next_vec,gamma=0.9):
        q_values = self.net.forward(state_vec)
        next_q = self.net.forward(next_vec)
        target = q_values[:]
        a_idx = ACTIONS.index(action)
        target[a_idx] = reward + gamma*max(next_q)
        self.net.train(target)

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.clear()

    height = GRID_SIZE + 2
    width = GRID_SIZE + 2
    win = curses.newwin(height, width, 0, 0)
    win.nodelay(True)
    win.keypad(True)

    game = SnakeGame()
    agent = Agent(input_size=8, hidden_size=32, output_size=4)  # bigger hidden layer
    episode = 0
    step_count = 0
    global MOVE_DELAY

    try:
        while True:
            game.reset()
            state_vec = agent.get_state_vector(game)
            total_reward = 0

            while not game.done:
                # Check for dynamic speed keys
                key = win.getch()
                if key == ord('+'):
                    MOVE_DELAY = max(0.001, MOVE_DELAY / 2)
                elif key == ord('-'):
                    MOVE_DELAY = min(1.0, MOVE_DELAY * 2)

                action = agent.choose_action(state_vec)
                reward = game.step(action)
                next_vec = agent.get_state_vector(game)
                agent.train_step(state_vec, action, reward, next_vec)
                state_vec = next_vec
                total_reward += reward

                agent.decay_epsilon()
                step_count += 1

                # Render the game
                game.render(win)
                stdscr.addstr(height + 1, 0,
                              f"Episode: {episode}  Reward: {total_reward}  Epsilon: {agent.epsilon:.3f}  Delay: {MOVE_DELAY:.3f}   ")
                stdscr.refresh()

                time.sleep(MOVE_DELAY)

            episode += 1

    except KeyboardInterrupt:
        curses.endwin()
        print("\nTraining stopped by user.")
        print(f"Episodes completed: {episode}")


if __name__ == "__main__":
    curses.wrapper(main)
localhost:~# 
