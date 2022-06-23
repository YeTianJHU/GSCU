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

The environment we use for Kuhn Poker is [OpenSpiel](https://github.com/deepmind/open_spiel), and the environment we use for pretator prey is the Multi-agent Particle Environment. 
To install OpenSpiel 
```
    python3 -m pip install open_spiel
```
To install the Multi-agent Particle Environment, please follow https://github.com/openai/multiagent-particle-envs 


### Directory Structure
```
    .
    ├── kuhn_poker                      # environment Kuhn Poker
    │   ├── embedding_learning          # Policy2Emb training
    │   ├── conditional_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for Policy2Emb/RL/opponents 
    │   └── utils                       # helper functions and environment config
    ├── predator_prey                   # environment Predator Prey
    │   ├── embedding_learning          # Policy2Emb training
    │   ├── conditional_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for Policy2Emb/RL/opponents 
    │   ├── multiagent                  # Multi-agent Particle Environment scenarios
    │   └── utils                       # helper functions and environment config
    ├── LICENSE
    └── README.md
```

### Running
The running protocol for Kuhn Poker and Predator Prey are the same. Here we take Kuhn Poker environment for example:
* To generate the training data for the Policy2Emb
```
    cd GSCU/kuhn_poker/embedding_learning
    python data_generation.py 
```
* To train the Policy2Emb model using the data generated
```
    cd GSCU/kuhn_poker/embedding_learning
    python train_vae.py 
```
* To train the conditional RL model once the Policy2Emb model is ready
```
    cd GSCU/kuhn_poker/conditioned_RL
    python train_conditional_rl.py -e <ENCODER_NAME>
```
* To run the online test for seen/unseen/mix settings
```
    cd GSCU/kuhn_poker/online_test
    python online_adaption.py -o SETTING -e <ENCODER_NAME> -d <DECODER_NAME> -r <CONDITIONAL_RL_NAME>
```
* We provided 5 opponent initial points with ID 1~5. To run the online test for adaptive settings
```
    cd GSCU/kuhn_poker/online_test
    python online_adaption_opponent_adaptive.py -oid OPPONENT_ID -e <ENCODER_NAME> -d <DECODER_NAME> -r <CONDITIONAL_RL_NAME>
```
Pre-trained weights for encoder/decoder/RL models are also provided.

### Citation
```
@inproceedings{fu2022greedy,
   title={Greedy when Sure and Conservative when Uncertain about the Opponents},
   author={Fu, Haobo and Tian, Ye and Yu, Hongxiang and Liu, Weiming and Wu, Shuang and Xiong, Jiechao and When, Ying and Li, Kai and Xing, Junliang and Fu, Qiang and Yang, Wei},
   booktitle={International Conference on Machine Learning},
   year={2022},
   organization={PMLR}
}
```

### Acknowledgements
For the partially observable setting in Predator Prey environment we refer to [LIAM](https://github.com/uoe-agents/LIAM). 