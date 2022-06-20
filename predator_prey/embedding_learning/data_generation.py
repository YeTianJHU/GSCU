import pickle
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time

from multiagent.environment import MultiAgentEnv
from multiagent.mypolicy import *

import multiagent.scenarios as scenarios


N_ADV = 3

# ramdomly generate 3 adv policies and a agent 
def generate_policies(env, adv_pool,agent_pool,adv_path):
    selected_advs_ids = np.random.choice(range(0, len(adv_pool)), size=N_ADV, replace=True)
    selected_agent_ids = np.random.choice(range(0, len(agent_pool)), size=1, replace=True)

    adv_policies = []
    agent_policies = []
    for idx, adv_id in enumerate(selected_advs_ids):
        policy = get_policy_by_name(env,adv_pool[adv_id],idx,adv_path[adv_id])
        adv_policies.append(policy)

    agent_policies = [eval(agent_pool[selected_agent_ids[0]] + "(env," + str(N_ADV) +")")]
    policies = adv_policies + agent_policies

    return policies, selected_advs_ids

def get_all_adv_policies(env,adv_pool,adv_path,agent_index):
    all_policies = []
    for adv_id in range(len(adv_pool)):
        policy = get_policy_by_name(env,adv_pool[adv_id],agent_index,adv_path[adv_id])
        all_policies.append(policy)
    return all_policies

def get_policy_by_name(env,policy_name,agent_index,param_path):
    if param_path is None:
        return eval(policy_name + "(env," + str(agent_index) +")")
    return eval(policy_name + "(env," + str(agent_index) + ","+"'" + str(param_path) + "'"+")")
    

def main(args):
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        shared_viewer=False, discrete_action=True)
    # render call to create viewer window (necessary only for interactive policy)
    # env._render()

    adv_pool = ['PolicyN', 'PolicyEA', 'PolicyW', 'PolicyS']
    adv_path = [None] * len(adv_pool)
    agent_pool = ['AgentPolicy']

    all_policies_idx0 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=0)
    all_policies_idx1 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=1)
    all_policies_idx2 = get_all_adv_policies(env,adv_pool,adv_path,agent_index=2)
    all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]

    print (all_policies_idx0)

    data_s = []
    data_a = []
    data_i = []

    n_step = 0
    n_sample = 0

    n_same_output = 0
    n_all_output = 0

    for e in range(args.eposides):
        policies, adv_ids = generate_policies(env, adv_pool,agent_pool,adv_path)

        # execution loop
        obs_n = env._reset()
        steps = args.steps
        for st in range(steps):
            start = time.time()
            # query for action from each agent's policy
            act_n = []
            for i, policy in enumerate(policies):
                act = policy.action(obs_n[i])
                act_n.append(act)
                this_action = []
                # collect the obs/a/policy_index for adv policies
                if i < N_ADV:
                    # simulate the action for all advs in the pool
                    for adv_id, sudo_policy in enumerate(all_policies[i]):
                        sudo_act = sudo_policy.action(obs_n[i])
                        data_s.append(obs_n[i])
                        data_a.append(sudo_act)
                        data_i.append(adv_id)

                        this_action.append(sudo_act)
                        n_sample += 1

                    result = all(np.argmax(element) == np.argmax(this_action[0]) for element in this_action)
                    if result:
                        n_same_output += 1
                    n_all_output += 1

            # step environment
            obs_n, reward_n, done_n, _ = env._step(act_n)
            n_step += 1

        if e % (args.eposides//10) == 0:
            print ('n eposides',e+1)
            print ('n steps',n_step)
            print ('n sample',n_sample)
            print ('n all_output',n_all_output)
            print ('n same_output',n_same_output)

            # display rewards
            print("Current step", st+1)
            for agent in env.world.agents:
                print("reward: %0.3f" % env._get_reward(agent))

    vae_data = {
        'data_s': data_s,
        'data_a': data_a,
        'data_i': data_i}

    pickle.dump(vae_data, 
        open('../data/vae_data_simple_tag_' + args.version + '.p', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    parser.add_argument('-st', '--steps', default=50, type=int, help='Num of steps in a single run')
    parser.add_argument('-ep', '--eposides', default=1000, type=int, help='Num of eposides for a fix set of policies')
    parser.add_argument('-v', '--version', default='v0')
    args = parser.parse_args()


    main(args)

