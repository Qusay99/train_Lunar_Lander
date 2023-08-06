# train_Lunar_Lander
This repository contains the implementation of DQN (Deep Q-Network) and DDQN (Double Deep Q-Network) agents to solve the LunarLander-v2 environment provided by [OpenAI Gym](https://www.gymlibrary.dev/index.html#).

Here are the main steps involved in the process:

1. Import necessary libraries: The code starts by importing the required libraries, including NumPy, pandas, gym, PyTorch, and others.

2. Create the LunarLander Environment: The code creates the LunarLander environment using OpenAI Gym.

3. Implement a Random Agent: The code defines a random agent that takes random actions in the environment. This is done to establish a baseline performance to compare with the trained agents.

4. Define the QNetwork class: The QNetwork is a deep neural network implemented using PyTorch. It serves as the function approximator for the Q-values in the Q-learning algorithm.

5. Define the ReplayBuffer class: The ReplayBuffer is used to store experiences (state, action, reward, next state, done) during agent interactions with the environment. It enables the agent to use experience replay for more stable learning.

6. Define the DQNAgent class: The DQNAgent implements the DQN algorithm, which includes the Q-learning update, experience replay, and epsilon-greedy exploration strategy.

7. Define the DDQNAgent class: The DDQNAgent extends the DQNAgent and implements the Double DQN algorithm, which uses separate networks for action selection and target Q-value estimation, reducing overestimation bias.

8. Set hyperparameters: The code sets various hyperparameters, such as the buffer size, batch size, discount factor, learning rate, update frequency, and epsilon values for exploration.

9. Train the DQN agent: The code trains the DQN agent in the LunarLander environment using the Q-learning algorithm with experience replay.

10. Train the DDQN agent: Similarly, the code trains the DDQN agent using the Double DQN algorithm.

11. Evaluate the trained agents: The code evaluates the performance of the trained DQN and DDQN agents by running episodes in the environment with the learned policies.

12. Visualize the training progress: The code plots the scores achieved during training to visualize the performance improvement over episodes.

13. Save the trained models: The trained models (DQN and DDQN) will be saved after finishing the training of each agent.

14. Test the agents: The code loads the trained models and tests the agents by running episodes in the environment.

## Requirements

You can install all the required dependencies by using the provided requirements.txt file.
To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

## Training and Testing the Agents
The training of both DQN and DDQN agents is implemented in the **main.ipynb** notebook. To train the agents, open the main.ipynb and run the notebook. After the training of each agent, the trained model will be saved in the root folder.

### Hyperparameters

You can modify the hyperparameters for training the agents in the train_agents.py script. The key hyperparameters include:

- BUFFER_SIZE: Size of the replay memory buffer.
- BATCH_SIZE: Number of experiences to sample from memory in each training step.
- GAMMA: Discount factor for calculating the target Q-values.
- TAU: Soft update parameter for updating the target network in DDQN.
- LR: Learning rate for the Q-network optimizer.
- UPDATE_EVERY: Frequency of updating the Q-network.

## Sources
- https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
- https://www.katnoria.com/nb_dqn_lunar/