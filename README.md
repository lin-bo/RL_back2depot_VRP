# RL_back2depot_VRP

## Intro
This is the codebase of an ongoing research project on using reinforcement learning to learn the return-to-depot policy for solving VRP. We propose a hierarchical reinforcement learning (RL) architecture that decomposes the VRP solution task into two sub-tasks to make the return-to-depot and the routing decisions, separately. Specifically, at each decision epoch, a returning RL agent first determines if the vehicle should return to the depot or not based on the current system state. If the agent decides not to return, a routing agent then determines the next customer to visit. The returning agent captures the customer assignment policy for solving VRP, enabling us to re-use the ML models trained for other routing problems. The proposed approach can be regarded as a transfer learning technique that helps to generalize pre-trained routing models to different problem variants.

## Structure
 - attention_model: this is the codebase developed by Kool et al. (2019). We re-use their pre-trained VRP and TSP models as the routing agents in our solution framework
 - pretrained: pre-trained returning agents for different types of VRP instances
 - prob: instance generator
 - res: experimental results
 - solver: baseline VRP solvers
 - utils: utility functions
 
 ## References
 Kool, W., Van Hoof, H. and Welling, M., 2018. Attention, learn to solve routing problems!. arXiv preprint arXiv:1803.08475.
