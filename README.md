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
    │   ├── conditional_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for OI-PEL/RL/opponents 
    │   └── utils                       # helper functions and environment config
    ├── predator_prey                   # environment Predator Prey
    │   ├── embedding_learning          # OI-PEL training
    │   ├── conditional_RL              # conditional RL training 
    │   ├── online_test                 # online bayesian adaption for seen/unseen/mix/adaptive settings 
    │   ├── data                        # data for embedding learning and online test sequences 
    │   ├── model_params                # model parameters for OI-PEL/RL/opponents 
    │   ├── multiagent                  # Multi-agent Particle Environment scenarios
    │   └── utils                       # helper functions and environment config
    ├── LICENSE
    └── README.md
```

### Running
The running method for kuhn poker and predator prey are the same. Here we take kuhn poker environment for example:
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
    python train_conditional_rl.py -e <ENCODER_NAME>
```
* To run the online test for seen/unseen/mix settings
```
    cd GSCU/kuhn_poker/online_test
```
For seen:
```
    python online_adaption.py -o seen -e <ENCODER_NAME> -d <DECODER_NAME> -r <RL_NAME>
```
For unseen:
```
    python online_adaption.py -o unseen -e <ENCODER_NAME> -d <DECODER_NAME> -r <RL_NAME>
```
For mix:
```
    python online_adaption.py -o mix -e <ENCODER_NAME> -d <DECODER_NAME> -r <RL_NAME>
```
* To run the online test for adaptive settings
```
   python online_adaption_opponent_adaptive.py -oid OPPONENT_ID -e <ENCODER_NAME> -d <DECODER_NAME> -r <RL_NAME>
```
We also provide pre-trained weight for encoder/decoder/RL models. 

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