# Offroad Autonomous Navigation 

Implementation of paper: [Uncertainty-aware hybrid paradigm of nonlinear MPC and model-based RL for offroad navigation: Exploration of transformers in the prediction model](https://arxiv.org/abs/2310.00760)

In this work, we investigate a hybrid control scheme that combines nonlinear model predictive control (MPC) and model-based reinforcement learning (RL) for the navigational planning of an RC car across off-road, unstructured terrains. Our work builds on [BADGR](https://github.com/gkahn13/badgr), by exploring the substitution of LSTM modules with Transformers to enhance environmental modeling capabilities. Addressing uncertainty within the system, we train an ensemble of prediction models and estimate the mutual information between model weights and outputs. This facilitates dynamic horizon planning through the introduction of variable speeds. 

We also incorporate a nonlinear MPC controller that accounts for the intricacies of the vehicle model. The model-based RL planner produces steering angles and quantifies inherent uncertainty. At the same time, the nonlinear MPC suggests optimal throttle settings, striking a balance between goal attainment speed and managing model uncertainty. Our approach excels in handling complex environmental challenges and integrates the vehicle's kinematic model to enhance decision-making.

Here are a few samples of our dataset from different trials: 
![repo](https://github.com/FARAZLOTFI/offroad_autonomous_navigation/assets/44290848/6cbdb552-d60f-4f75-9b79-79533a573ada)


The link to download our dataset <--!https://mcgill-my.sharepoint.com/:u:/g/personal/khalil_virji_mail_mcgill_ca/EeqSokfjlP5ItDJu1XBu6GUB3WQVLOMtolhk3442upqUvw?e=rfLHmd-->

# Code Structure
* classifier/
  * The training and evaluation pipleline for comparing state-of-the-art classification models on our dataset
* dataset_preparation/
  * The annotation tool used for manually annotating our dataset
* extract_dataset/
  * [ROS node](https://github.com/FARAZLOTFI/offroad_autonomous_navigation/blob/main/extract_dataset/rc_subscriber_node.py) for extracting our dataset from collected rosbags. The code and plots used to analyze the dataset's feature distribution are also included in this directory. 
* models/
  * BADGR, LSTM, and Transformer networks
* results/
  * Results for LSTM, Transformer, and Speed studies
* metrics.py - metrics used for performance evaluation. Based on [PyTorch Ignite Metrics](https://pytorch.org/ignite/index.html)
* traj_planner_[train/eval].py - training, and evaluation pipelines for the trajectory planner. Replace `seq_encoder = LSTMSeqModel(...)` with `seq_encoder = TransformerSeqModel(...)` to switch between LSTM and Transformer modules.
* traj_planner_helpers.py - helper functions such as input preparation, and loss function calculations used in both the trajectory planner training and evaluation pipelines.
* traj_planner_visualize.py - pipeline to easily visualize model predictions for a given input. 
* MHE_MPC
  *  hybrid_controller.py - Implementation of the hybrid planner
  *  System Identification - The moving horizon estimator to estimate the states and parameters of the vehicle
* Uncertainty Branch
  * This branch has the same structure as the main branch, while incorporating the uncertainty in it.    
