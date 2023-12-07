import pygame
import random
from sys import exit
ROWS = 8
COLS = 8
WIN_POS = (7, 7)
LOSE_POS = (7, 0)
CELL_SIZE = 80
START_POS = (0, 4)
clock = pygame.time.Clock()

class GridEnv:
    

    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((ROWS * CELL_SIZE, COLS * CELL_SIZE))
        self.states = [pygame.Vector2(x, y) for x in  range(ROWS) for y in range(COLS)]
        self.player = pygame.Vector2(START_POS)
        self.ACTIONS = [0, 1, 2, 3]
        self.action_space = len(self.ACTIONS)
        self.state_space = len(self.states)
        pygame.key.set_repeat()
        
    
    def reset(self):
        self.player = pygame.Vector2(START_POS)
        
        return (self.player.x, self.player.y)
    
    
    def draw(self):
        for point in self.states:
            if point == WIN_POS:
                color = "blue"
            elif point == LOSE_POS:
                color = "red"
            else:
                color = "white"
            rect = pygame.Rect(point.x*CELL_SIZE, point.y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.display_surface, color, rect)
        player_rect = pygame.Rect(self.player.x * CELL_SIZE, self.player.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.display_surface, "green", player_rect)
    
    def draw_grid(self):
        for row in range(ROWS):
            pygame.draw.line(self.display_surface, "black", (row * CELL_SIZE, 0), (row * CELL_SIZE, ROWS  * CELL_SIZE))
            pygame.draw.line(self.display_surface, "black", (0, row * CELL_SIZE), (ROWS  * CELL_SIZE, row * CELL_SIZE))
            
    def get_state(self):
        return (self.player.x, self.player.y)
    def step(self, action) -> tuple[int,tuple[int, int], bool]:
        #if event.type == pygame.KEYDOWN:
        self.display()
        prev_state = (self.player.x, self.player.y)
        if action == 0 and self.player.y > 0:
            self.player.y -= 1
        if action == 1 and self.player.y < ROWS - 1:
            self.player.y += 1
        if action == 2 and self.player.x > 0:
            self.player.x -= 1
        if action == 3 and self.player.x < COLS - 1:
            self.player.x += 1
        return self.reward(),self.get_state(), self.game_end()
        
    
    def reward(self):
        if self.player == WIN_POS:
            reward = 1
        elif self.player == LOSE_POS:
            reward = -1
        else:
            reward = 0
        return reward
    
    def game_end(self):
        if self.player == WIN_POS:
            self.win = True
            return True
        if self.player == LOSE_POS:
            self.win = False            
            return True
        return False
    
    def display(self):
        #while True:
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            #self.step()
        # if self.game_end():
        #     self.reset()
            # pygame.quit()
            # exit()
        self.draw()
        self.draw_grid()
        pygame.display.update()
        
        

def create_Qtable(env) -> dict:
    Q = {}
    for state in env.states:
        Q[(state[0], state[1])] = [0 for _ in range(env.action_space)]
        
    return Q
        


def sarsa(n_iters:int, env:GridEnv, action_space:int,  gamma:float, lr:float, eps:float=0.5)->None:
    """Actually it's not exactly sarsa since in the evaluation I use the greedy policy

    Args:
        n_iters (int): the number of iteration
        env (_type_): the environment
        action_space (int): the environment's action space
        state_space (int): the environment's state space
        gamma (float): the discount factor
        lr (float): Learning rate
        eps (float, optional): Since the epsilon greedy policy is used a value for that. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    
    Q = create_Qtable(env)
    for state in env.states:
        Q[(state[0], state[1])] = [0 for x in range(action_space)]
    
    def select_action(s):
        # print(eps)
        # print(type(eps))
        rand = random.random()
        action = Q[(s[0], s[1])].index(max(Q[(s[0], s[1])])) if rand > eps else random.choice([0, 1, 2, 3])
        #print(action)
        return action
    for episode in range(n_iters):
        state = env.reset()
        action = select_action(state)
        while True:

            #clock.tick(10)
            

            reward, next_state, ended = env.step(action)
            
            if ended:
                # print(f"s{s}")
                # print(f"next_state{next_state}")
                #print("here?")
                Q[(state[0], state[1])][action] = Q[(state[0], state[1])][action] +  lr * reward
                break
            next_action = select_action(next_state)
            Q[(state[0], state[1])][action] = Q[(state[0], state[1])][action] + lr * (reward + gamma * Q[(next_state[0], next_state[1])][next_action] - Q[(state[0], state[1])][action])
            state = next_state
            action = next_action
    #print(Q)
    state = env.reset()
    while True:
        clock.tick(10)
        
        action = Q[(state[0], state[1])].index(max(Q[(state[0], state[1])]))
        print(action)
        r, n_s, d = env.step(action)
        if d:
            print(env.win)
            break
        state = n_s
                
            
def q_learning(n_iters:int, env:GridEnv, action_space:int, gamma:float, lr:float, eps:float = 0.5  ) -> None:
    """Q-learning, difference from sarsa, in sarsa the updates are done according to the Policy, which in this implementation
    is the epsilon greedy policy, with out reducing the epsilon value
    while here in Q-learning the Q-values are updated, based on the value of the best next_state
    so the one that has the highest value

    Args:
        n_iters (int): _description_
        env (_type_): _description_
        action_space (int): _description_
        state_space (int): _description_
        gamma (float): _description_
        lr (float): _description_
        eps (float, optional): _description_. Defaults to 0.5.
    """
    Q = create_Qtable(env)
    
    
    def select_action(s) -> int:
        rand = random.random()
        
        action = random.choice([a for a in range(action_space)]) if rand <= eps else Q[(s[0], s[1])].index(max(Q[(s[0], s[1])]))
        #print(action)
        return action    
    
    
    for iter in range(n_iters):
        state = env.reset()
        
        while True:
            #clock.tick(10)
            action = select_action(state)
            
            reward, next_state, done = env.step(action)
            
            if done:
                Q[state][action] = Q[state][action] + lr * reward
                break
            Q[state][action] = Q[state][action] + lr * \
                (reward  + gamma * Q[next_state][Q[next_state].index(max(Q[next_state]))] - Q[state][action])
            state = next_state
    #print(Q)
    s  = env.reset()
    while True:
        clock.tick(10)
        print(action)
        action = Q[state].index(max(Q[state]))
        
        _, next_state, done = env.step(action)
        
        if done:
            print(env.win)
            break
        state = next_state
        
        
            
            
            
            
            
    
    

            

if __name__ == "__main__":
    g = GridEnv()
    
    sarsa(100, g, g.action_space, 0.1, 0.1, 0.4)
    q_learning(1000, g, g.action_space, 0.1, 0.1, 0.8)
    
    # episodes = []
    # n_iters = 10
    # for i in range(n_iters):
    #     ep_sum = 0
    #     while True:
    #         clock.tick(5)
    #         g.step(random.choice([0, 1, 2, 3]))
    #         ep_sum =+ g.reward()
    #         if g.game_end():
    #             break
    #     episodes.append(ep_sum)
    #     g.reset()
        
    #print(episodes)

    