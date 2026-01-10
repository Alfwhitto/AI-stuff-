import random
import time
import curses
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# ---------------------
# Constants
# ---------------------
GRID_SIZE = 10
MOVE_DELAY = 0.1
ACTIONS = ['w','a','s','d']
OPPOSITE = {'w':'s','s':'w','a':'d','d':'a'}

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9985  # Faster decay for faster exploitation
GAMMA = 0.99  # Higher discount factor for planning ahead
LR = 0.001  # Better learning rate for faster convergence
BATCH_SIZE = 64  # Balanced batch size
MEMORY_SIZE = 50000  # Larger replay buffer
TARGET_UPDATE = 100  # More frequent updates for faster learning
GRADIENT_CLIP = 1.0  # Clip gradients for stability

REWARD_FOOD = 200  # Increased incentive to eat
REWARD_CLOSER = 15  # Increased step-wise progress reward
REWARD_FARTHER = -20  # Reduced penalty for moving away
REWARD_DEATH = -100  # Lower death penalty
REWARD_DIRECTION = 5  # Reward moving toward food
REWARD_TIGHT_SPACE = -40  # Reduced tight space penalty
REWARD_SURVIVAL = 1  # Small reward for each step alive to encourage longer games

# ---------------------
# Snake Game
# ---------------------
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(5,5),(5,4),(5,3)]
        self.direction = 'd'
        self.done = False
        self.spawn_food()
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = (random.randint(1, GRID_SIZE), random.randint(1, GRID_SIZE))
            if self.food not in self.snake:
                return

    def step(self, action):
        if action == OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        hy,hx = self.snake[0]
        if action=='w': new_head = (hy-1,hx)
        elif action=='s': new_head = (hy+1,hx)
        elif action=='a': new_head = (hy,hx-1)
        else: new_head = (hy,hx+1)

        old_dist = self.distance_to_food()

        if (new_head[0]==0 or new_head[0]==GRID_SIZE+1 or
            new_head[1]==0 or new_head[1]==GRID_SIZE+1 or
            new_head in self.snake):
            self.done = True
            return self.get_state(), REWARD_DEATH, True

        self.snake.insert(0,new_head)
        reward = 0

        if new_head == self.food:
            self.spawn_food()
            reward += REWARD_FOOD
        else:
            self.snake.pop()
            reward += REWARD_SURVIVAL  # Reward for staying alive
            new_dist = self.distance_to_food()
            reward += REWARD_CLOSER if new_dist < old_dist else REWARD_FARTHER

        # Directional reward
        fy,fx = self.food
        if action=='w' and fy<hy: reward += REWARD_DIRECTION
        if action=='s' and fy>hy: reward += REWARD_DIRECTION
        if action=='a' and fx<hx: reward += REWARD_DIRECTION
        if action=='d' and fx>hx: reward += REWARD_DIRECTION

        # Immediate danger
        danger_w = 1 if hy-1==0 or (hy-1,hx) in self.snake[3:] else 0
        danger_s = 1 if hy+1==GRID_SIZE+1 or (hy+1,hx) in self.snake[3:] else 0
        danger_a = 1 if hx-1==0 or (hy,hx-1) in self.snake[3:] else 0
        danger_d = 1 if hx+1==GRID_SIZE+1 or (hy,hx+1) in self.snake[3:] else 0
        if action=='w' and danger_w: reward -= REWARD_FARTHER*3
        if action=='s' and danger_s: reward -= REWARD_FARTHER*3
        if action=='a' and danger_a: reward -= REWARD_FARTHER*3
        if action=='d' and danger_d: reward -= REWARD_FARTHER*3

        # Penalty for heading into tight space
        free_ahead = self.count_free_ahead(action)
        if free_ahead <= 2:
            reward += REWARD_TIGHT_SPACE

        return self.get_state(), reward, False

    def distance_to_food(self):
        hy,hx = self.snake[0]
        fy,fx = self.food
        return abs(hy-fy)+abs(hx-fx)

    # Count free squares ahead in direction
    def count_free_ahead(self,direction):
        hy,hx = self.snake[0]
        distance = 0
        while True:
            if direction=='w': pos = (hy-distance-1,hx)
            elif direction=='s': pos = (hy+distance+1,hx)
            elif direction=='a': pos = (hy,hx-distance-1)
            else: pos = (hy,hx+distance+1)
            if pos[0]==0 or pos[0]==GRID_SIZE+1 or pos[1]==0 or pos[1]==GRID_SIZE+1:
                return distance
            if pos in self.snake[3:]:
                return distance
            distance += 1

    def get_state(self):
        hy,hx = self.snake[0]
        fy,fx = self.food
        dy = 0 if fy==hy else (1 if fy>hy else -1)
        dx = 0 if fx==hx else (1 if fx>hx else -1)
        dy_dist = (fy-hy)/GRID_SIZE
        dx_dist = (fx-hx)/GRID_SIZE
        
        # Vectorized danger checks for all 4 directions
        danger_w = 1 if hy-1==0 or (hy-1,hx) in self.snake[3:] else 0
        danger_s = 1 if hy+1==GRID_SIZE+1 or (hy+1,hx) in self.snake[3:] else 0
        danger_a = 1 if hx-1==0 or (hy,hx-1) in self.snake[3:] else 0
        danger_d = 1 if hx+1==GRID_SIZE+1 or (hy,hx+1) in self.snake[3:] else 0

        # Cached snake set for faster lookups
        snake_set = set(self.snake[3:])
        
        # Tail distance in each direction (optimized with early exits)
        def tail_dist(dir):
            distance = 1
            while distance <= GRID_SIZE:
                if dir=='w': pos=(hy-distance,hx)
                elif dir=='s': pos=(hy+distance,hx)
                elif dir=='a': pos=(hy,hx-distance)
                else: pos=(hy,hx+distance)
                if pos[0]==0 or pos[0]==GRID_SIZE+1 or pos[1]==0 or pos[1]==GRID_SIZE+1: 
                    return 0
                if pos in snake_set: 
                    return 0
                distance+=1
            return 1
        
        danger_forward = tail_dist(self.direction)
        left_map = {'w':'a','a':'s','s':'d','d':'w'}
        right_map = {'w':'d','d':'s','s':'a','a':'w'}
        danger_left = tail_dist(left_map[self.direction])
        danger_right = tail_dist(right_map[self.direction])
        danger_back = tail_dist(OPPOSITE[self.direction])

        return np.array([dy,dx,dy_dist,dx_dist,danger_w,danger_s,danger_a,danger_d,
                         danger_forward,danger_left,danger_right,danger_back],dtype=np.float32)

    def render(self,win):
        win.clear()
        win.border()
        for y,x in self.snake: win.addch(y,x,'#')
        fy,fx = self.food
        win.addch(fy,fx,'*')
        win.refresh()

# ---------------------
# DQN & Agent
# ---------------------
class DQN(nn.Module):
    def __init__(self,input_size=12,hidden_size=256,output_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2,output_size)
        )
    def forward(self,x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=LR,betas=(0.9,0.999))
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self,state):
        self.steps_done+=1
        if random.random()<self.epsilon:
            return random.randint(0,3)
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(state_t).argmax())

    def store_transition(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def optimize(self):
        if len(self.memory)<BATCH_SIZE: return
        batch = random.sample(self.memory,BATCH_SIZE)
        states = torch.tensor(np.array([b[0] for b in batch]),dtype=torch.float32).to(self.device)
        actions = torch.tensor([b[1] for b in batch]).unsqueeze(1).to(self.device)
        rewards = torch.tensor([b[2] for b in batch],dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]),dtype=torch.float32).to(self.device)
        dones = torch.tensor([b[4] for b in batch],dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1,actions)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA*(1-dones)*next_q

        loss = nn.SmoothL1Loss()(q_values, target_q)  # Huber loss is more robust
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP)
        self.optimizer.step()
        
        if self.epsilon>EPSILON_MIN: 
            self.epsilon*=EPSILON_DECAY

# ---------------------
# Main Loop
# ---------------------
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.clear()
    height = GRID_SIZE+2
    width = GRID_SIZE+2
    win = curses.newwin(height,width,0,0)
    win.nodelay(True)
    win.keypad(True)

    game = SnakeGame()
    agent = Agent()
    episode=0
    global MOVE_DELAY

    try:
        while True:
            state = game.reset()
            total_reward=0
            while not game.done:
                key = win.getch()
                if key==ord('+'): MOVE_DELAY=max(0.001,MOVE_DELAY/2)
                elif key==ord('-'): MOVE_DELAY=min(1.0,MOVE_DELAY*2)

                action = agent.select_action(state)
                next_state,reward,done = game.step(ACTIONS[action])
                agent.store_transition(state,action,reward,next_state,done)
                agent.optimize()
                state = next_state
                total_reward += reward

                if agent.steps_done % TARGET_UPDATE == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

                game.render(win)
                stdscr.addstr(height+1,0,
                    f"Episode:{episode} Reward:{total_reward} Epsilon:{agent.epsilon:.3f} Delay:{MOVE_DELAY:.3f}   ")
                stdscr.addstr(height+2,0,f"State: {state}")
                stdscr.refresh()
                time.sleep(MOVE_DELAY)

            episode+=1

    except KeyboardInterrupt:
        curses.endwin()
        print("\nTraining stopped by user.")
        print(f"Episodes completed: {episode}")

if __name__=="__main__":
    curses.wrapper(main)
