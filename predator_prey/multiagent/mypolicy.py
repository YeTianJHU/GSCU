import numpy as np
import random

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from conditional_RL.ppo_model import PPO

class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()
    def check_bound(self, p_pos):
        # check if the given pos is inside the screen
        if abs(p_pos[0]) > 1.0 or abs(p_pos[1]) > 1.0:
            return True
        return False
    def action_outside_bound(self, p_pos):
        # go back to screen if outside
        delta_pos = - p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))
        if self.env.discrete_action_input:
            u = np.argmin(dist) + 1
        else:
            u = np.zeros(5)
            idx = np.argmin(dist) + 1
            u[idx] += 1
        return np.concatenate([u, np.zeros(2)])

class PolicyW(Policy):
    # adv prefers to go west (-1, 0)
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyW, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = 1
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_u[1] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyEA(Policy):
    # adv prefers to go east (1, 0)
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyEA, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = 2
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_u[2] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyS(Policy):
    # adv prefers to go south (0, -1)
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyS, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = 3
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_u[3] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyN(Policy):
    # adv prefers to go west (0, 1)
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyN, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = 4
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_u[4] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyStay(Policy):
    # adv prefers to stay
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyStay, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = 0
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_u[0] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyNO(Policy):
    # adv prefers to go obstacle
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyNO, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        delta_pos_landmarks = [entity.state.p_pos - agent.state.p_pos for entity in self.env.world.landmarks]
        if np.sum(np.square(delta_pos_landmarks[0])) < np.sum(np.square(delta_pos_landmarks[1])):
            delta_pos_ob = delta_pos_landmarks[0]
        else:
            delta_pos_ob = delta_pos_landmarks[1]
        dist_ob = np.zeros(4)
        dist_ob[0] = np.sqrt(np.sum(np.square(delta_pos_ob+(-1,0))))
        dist_ob[1] = np.sqrt(np.sum(np.square(delta_pos_ob+(1,0))))
        dist_ob[2] = np.sqrt(np.sum(np.square(delta_pos_ob+(0,-1))))
        dist_ob[3] = np.sqrt(np.sum(np.square(delta_pos_ob+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = np.argmin(dist_ob) + 1
                u = np.argmax(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_idx = np.argmin(dist_ob) + 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
                greedy_u[greedy_idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyFO(Policy):
    # adv prefers to stay away from obstacle
    def __init__(self, env, agent_index, greedy_epsilon=0.8):
        super(PolicyFO, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.greedy_epsilon = greedy_epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        dist = np.zeros(4)
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))

        delta_pos_landmarks = [entity.state.p_pos - agent.state.p_pos for entity in self.env.world.landmarks]
        dist_ob = np.zeros(4)
        dist_ob[0] = np.sqrt(np.sum(np.square(delta_pos_landmarks[0]+(-1,0)))) + np.sqrt(np.sum(np.square(delta_pos_landmarks[1]+(-1,0))))
        dist_ob[1] = np.sqrt(np.sum(np.square(delta_pos_landmarks[0]+(1,0)))) + np.sqrt(np.sum(np.square(delta_pos_landmarks[1]+(1,0))))
        dist_ob[2] = np.sqrt(np.sum(np.square(delta_pos_landmarks[0]+(0,-1)))) + np.sqrt(np.sum(np.square(delta_pos_landmarks[1]+(0,-1))))
        dist_ob[3] = np.sqrt(np.sum(np.square(delta_pos_landmarks[0]+(0,1)))) + np.sqrt(np.sum(np.square(delta_pos_landmarks[1]+(0,1))))

        if self.check_bound(curadv.state.p_pos):
            return self.action_outside_bound(curadv.state.p_pos)
        else:
            if self.env.discrete_action_input:
                greedy_u = np.argmax(dist_ob) + 1
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                greedy_u = np.zeros(5)
                greedy_idx = np.argmax(dist_ob) + 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
                greedy_u[greedy_idx] += 1
            
            random_select = np.random.random()
            if random_select < self.greedy_epsilon:
                select_u = greedy_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 
    
class RandomPolicy(Policy):
    def __init__(self, env, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env

    def action(self, obs):
        # Randomly select an action
        if self.env.discrete_action_input:
            u = random.randint(1, 4)
        else:
            u = np.zeros(5)
            idx = random.randint(0, 4)
            u[idx] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class PolicyRL1(Policy):
    def __init__(self, env, agent_index, param_path = '../model_params/opponent/params_opp_init_1.pt'):
        super(PolicyRL1, self).__init__()
        self.env = env
        self.agent_index = agent_index
        state_dim = env.observation_space[agent_index].shape[0]
        action_dim = env.action_space[env.n-1].n
        hidden_dim = 128
        self.ppo_model = PPO(state_dim, hidden_dim, action_dim, 1e-5, 1e-4, 0.98)
        self.ppo_model.init_from_save(param_path)
    
    def action(self, obs):
        act,_,_ = self.ppo_model.select_action(obs, self.env.world.dim_c)
        return act

class PolicyRL2(Policy):
    def __init__(self, env, agent_index, param_path = '../model_params/opponent/params_opp_init_2.pt'):
        super(PolicyRL2, self).__init__()
        self.env = env
        self.agent_index = agent_index
        state_dim = env.observation_space[agent_index].shape[0]
        action_dim = env.action_space[env.n-1].n
        hidden_dim = 128
        self.ppo_model = PPO(state_dim, hidden_dim, action_dim, 1e-5, 1e-4, 0.98)
        self.ppo_model.init_from_save(param_path)
    
    def action(self, obs):
        act,_,_ = self.ppo_model.select_action(obs, self.env.world.dim_c)
        return act
        

class AgentPolicy(Policy):
    # Maximize the distance to the closest adv inside the screen with prob of 1-epsilon and move ramdomly with prob epsilon; 
    # avoiding going outside
    def __init__(self, env, agent_index, epsilon=0.2):
        super(AgentPolicy, self).__init__()
        self.env = env
        self.epsilon = epsilon
        self.agent_index = agent_index

    def action(self, obs):
        agent = self.env.world.agents[-1]
        agent_pos = agent.state.p_pos

        delta_pos_adv = []
        for i in range(self.env.n - 1):
            delta_pos_adv.append(self.env.agents[i].state.p_pos - agent.state.p_pos)
        dist_adv = np.zeros(4)
        dist_adv[0] = min([np.sqrt(np.sum(np.square(pos + (-1,0)))) for pos in delta_pos_adv])
        dist_adv[1] = min([np.sqrt(np.sum(np.square(pos + (1,0)))) for pos in delta_pos_adv])
        dist_adv[2] = min([np.sqrt(np.sum(np.square(pos + (0,-1)))) for pos in delta_pos_adv])
        dist_adv[3] = min([np.sqrt(np.sum(np.square(pos + (0,1)))) for pos in delta_pos_adv])

        if self.check_bound(agent_pos):
            return self.action_outside_bound(agent_pos)
        else:
            if np.random.random() < self.epsilon:
                if self.env.discrete_action_input:
                    u = random.randint(0, 4)
                else:
                    u = np.zeros(5)
                    idx = random.randint(0, 4)
                    u[idx] += 1.0
            else:
                # Randomly select an action
                if self.env.discrete_action_input:
                    u = np.argmax(dist_adv) + 1
                else:
                    u = np.zeros(5)
                    idx = np.argmax(dist_adv) + 1
                    u[idx] += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

        
class ConservativePolicy(Policy):
    # Maximize the distance to the closest adv inside the screen with prob of 1-epsilon and move ramdomly with prob epsilon; 
    # avoiding going outside
    def __init__(self, env, agent_index, epsilon=0.2):
        super(ConservativePolicy, self).__init__()
        self.env = env
        self.epsilon = epsilon
        self.agent_index = agent_index

    def action(self, obs):
        agent = self.env.world.agents[-1]
        agent_pos = agent.state.p_pos

        delta_pos_adv = []
        for i in range(self.env.n - 1):
            dist = self.env.agents[i].state.p_pos - agent.state.p_pos
            # partially observable
            if abs(dist[0]) <= 0.5 and abs(dist[1]) <= 0.5:
                delta_pos_adv.append(dist)
        
        if self.check_bound(agent_pos):
            return self.action_outside_bound(agent_pos)
        
        random_select = np.random.random()
        if not len(delta_pos_adv) or random_select < self.epsilon:
            if self.env.discrete_action_input:
                u = random.randint(0, 4)
            else:
                u = np.zeros(5)
                idx = random.randint(0, 4)
                u[idx] += 1.0
        else:
            dist_adv = np.zeros(4)
            dist_adv[0] = min([np.sqrt(np.sum(np.square(pos + (-1,0)))) for pos in delta_pos_adv])
            dist_adv[1] = min([np.sqrt(np.sum(np.square(pos + (1,0)))) for pos in delta_pos_adv])
            dist_adv[2] = min([np.sqrt(np.sum(np.square(pos + (0,-1)))) for pos in delta_pos_adv])
            dist_adv[3] = min([np.sqrt(np.sum(np.square(pos + (0,1)))) for pos in delta_pos_adv])

            if self.env.discrete_action_input:
                u = np.argmax(dist_adv) + 1
            else:
                u = np.zeros(5)
                idx = np.argmax(dist_adv) + 1
                u[idx] += 1

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class PolicyA(Policy):
    # adversary minimize its distance to agent
    def __init__(self, env, agent_index):
        super(PolicyA, self).__init__()
        self.env = env
        self.agent_index = agent_index
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        # Distance after taking u action 1,2,3,4
        dist = np.zeros(4)
        #dist[0] = np.sqrt(np.sum(np.square(delta_pos)))
        dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
        dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
        dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
        dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))
        
        if self.env.discrete_action_input:
            u = np.argmin(dist) + 1
        else:
            u = np.zeros(5)
            idx = np.argmin(dist) + 1
            u[idx] += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

class PolicyE(Policy):
    # epsilon-greedy policy
    def __init__(self, env, agent_index, epsilon=0.2):
        super(PolicyE, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.epsilon = epsilon
    
    def action(self, obs):
        agent = self.env.world.agents[-1]
        curadv = self.env.world.agents[self.agent_index]
        delta_pos = agent.state.p_pos - curadv.state.p_pos
        if np.random.random() < self.epsilon:
            if self.env.discrete_action_input:
                u = random.randint(1, 4)
            else:
                u = np.zeros(5)
                idx = random.randint(0, 4)
                u[idx] += 1.0
        else:
            dist = np.zeros(4)
            #dist[0] = np.sqrt(np.sum(np.square(delta_pos)))
            dist[0] = np.sqrt(np.sum(np.square(delta_pos+(-1,0))))
            dist[1] = np.sqrt(np.sum(np.square(delta_pos+(1,0))))
            dist[2] = np.sqrt(np.sum(np.square(delta_pos+(0,-1))))
            dist[3] = np.sqrt(np.sum(np.square(delta_pos+(0,1))))
            if self.env.discrete_action_input:
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                idx = np.argmin(dist) + 1
                u[idx] += 1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])