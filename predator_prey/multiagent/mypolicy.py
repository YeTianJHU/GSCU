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
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyW, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = 1
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = np.zeros(5)
                preference_u[1] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon:
                select_u = preference_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyE(Policy):
    # adv prefers to go east (1, 0)
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyE, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = 2
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = np.zeros(5)
                preference_u[2] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon:
                select_u = preference_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyS(Policy):
    # adv prefers to go south (0, -1)
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyS, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = 3
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = np.zeros(5)
                preference_u[3] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon:
                select_u = preference_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyN(Policy):
    # adv prefers to go north (0, 1)
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyN, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = 4
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = np.zeros(5)
                preference_u[4] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon:
                select_u = preference_u
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicySW(Policy):
    # adv prefers to go west with prob epsilon/2, go south with prob epsilon/2 
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicySW, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = []
                preference_u[0] = 1
                preference_u[1] = 3
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = [np.zeros(5), np.zeros(5)]
                preference_u[0][1] += 1
                preference_u[1][3] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon / 2:
                select_u = preference_u[0]
            elif random_select < self.preference_epsilon:
                select_u = preference_u[1]
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyNW(Policy):
    # adv prefers to go west with prob epsilon/2, go north with prob epsilon/2
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyNW, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = []
                preference_u[0] = 1
                preference_u[1] = 4
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = [np.zeros(5), np.zeros(5)]
                preference_u[0][1] += 1
                preference_u[1][4] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon / 2:
                select_u = preference_u[0]
            elif random_select < self.preference_epsilon:
                select_u = preference_u[1]
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicySE(Policy):
    # adv prefers to go east with prob epsilon/2, go south with prob epsilon/2
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicySE, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = []
                preference_u[0] = 2
                preference_u[1] = 3
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = [np.zeros(5), np.zeros(5)]
                preference_u[0][2] += 1
                preference_u[1][3] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon / 2:
                select_u = preference_u[0]
            elif random_select < self.preference_epsilon:
                select_u = preference_u[1]
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 

class PolicyNE(Policy):
    # adv prefers to go east with prob epsilon/2, go north with prob epsilon/2
    def __init__(self, env, agent_index, preference_epsilon=0.6):
        super(PolicyNE, self).__init__()
        self.env = env
        self.agent_index = agent_index
        self.preference_epsilon = preference_epsilon
    
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
                preference_u = []
                preference_u[0] = 2
                preference_u[1] = 4
                u = np.argmin(dist) + 1
            else:
                u = np.zeros(5)
                preference_u = [np.zeros(5), np.zeros(5)]
                preference_u[0][2] += 1
                preference_u[1][4] += 1
                idx = np.argmin(dist) + 1
                u[idx] += 1
            
            random_select = np.random.random()
            if random_select < self.preference_epsilon / 2:
                select_u = preference_u[0]
            elif random_select < self.preference_epsilon:
                select_u = preference_u[1]
            else:
                select_u = u
        return np.concatenate([select_u, np.zeros(self.env.world.dim_c)]) 


class ConservativePolicy(Policy):
    def __init__(self, env, agent_index, param_path = '../model_params/RL/params_pi.pt'):
        super(ConservativePolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index
        state_dim = env.observation_space[agent_index].shape[0]
        action_dim = env.action_space[env.n-1].n
        hidden_dim = 128
        self.ppo_model = PPO(state_dim, hidden_dim, action_dim, 0.0, 0.0, 0.99)
        self.ppo_model.init_from_save(param_path)
    
    def action(self, obs):
        act, act_index, act_prob = self.ppo_model.select_action(obs, self.env.world.dim_c)
        return act

