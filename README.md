# Greedy when Sure and Conservative when Uncertain about the Opponents (GSCU)
Repo for the Greedy when Sure and Conservative when Uncertain about the Opponents (ICML 2022).


### Requirements:
Our code is written in python 3.x. The following packages are needed:
```
- PyTorch (tested on 1.10.1)
- pandas
- joblib
- scipy
- pickle 
```

The environment we use for Kuhn Poker is OpenSpiel, and the environment we use for pretator prey is the Multi-agent Particle Environment. 
To install OpenSpiel 
```
    python3 -m pip install open_spiel
```
To install the Multi-agent Particle Environment, please follow https://github.com/openai/multiagent-particle-envs 


### Directory Structure
```
    .
    ├── kuhn_poker                      # environment Kuhn Poker
    │   ├── embedding_learning          # OI-PEL training
    │   ├── conditioned_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for OI-PEL/RL/opponents 
    │   └── utils                       # helper functions and environment config
    ├── predator_prey                   # environment Predator Prey
    │   ├── embedding_learning          # OI-PEL training
    │   ├── conditioned_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for OI-PEL/RL/opponents 
    │   └── multiagent                  # Multi-agent Particle Environment scenarios
    ├── LICENSE
    └── README.md
```

### Kuhn Poker
* To generate the training data for the OI-PEL
```
    cd GSCU/kuhn_poker/embedding_learning
    python data_generation.py
```
* To train the OI-PEL model using the data generated
```
    cd GSCU/kuhn_poker/VAE
    python train_vae.py
```
* To train the conditional RL model once the OI-PEL model is ready
```
    cd GSCU/kuhn_poker/conditioned_RL
    python train_conditional_rl.py
```
* To run the online test for seen/unseen/mix settings
```
    cd GSCU/kuhn_poker/online_test
```
For seen:
```
    python online_adaption.py -o seen
```
For unseen:
```
    python online_adaption.py -o unseen
```
For mix:
```
    python online_adaption.py -o mix
```
* To run the online test for adaptive settings
```
   python online_adaption_opponent_adaptive.py
```


### Predator Prey
* To train the OI-PEL model using the data generated
```
    cd GSCU/predator_prey/VAE
    python train_vae.py
```
* To train the conditional RL model once the OI-PEL model is ready
```
    cd GSCU/predator_prey/conditional_RL
    python train.py
```
* To run the online test for seen/unseen/mix settings
```
    cd GSCU/online_test
    python online_adaption.py
```
* To run long sequence for average performance
```
   python sequence_test.py
```
You can modify adv_type = 'seen' / 'unseen' / 'mix' to test on different settings. Note that for 'seen' and 'unseen', please load 'policy_vec_sequence_4.p' as policy vector, for 'mix', please load 'policy_vec_sequence_8.p' as policy vector
* To run the online test for adaptive settings
```
   python adaptive.py
```
