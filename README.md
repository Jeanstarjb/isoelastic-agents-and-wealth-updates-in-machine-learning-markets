# Isoelastic Agents and Wealth Updates in Machine Learning Markets

This repository provides an implementation of the research paper **"Isoelastic Agents and Wealth Updates in Machine Learning Markets"** by Amos Storkey, Jono Millin, and Krzysztof Geras. The paper explores the use of prediction markets as mechanisms for machine learning and demonstrates how agents with isoelastic utilities can be leveraged for model updates, resulting in improved prediction performance compared to traditional methods.

## Overview

### Core Concepts

The paper introduces the concept of **isoelastic utilities** in the context of prediction markets for machine learning. Here are the key ideas:

1. **Isoelastic Utility Functions**:
   Isoelastic utility functions are a class of utility functions defined as:
   ```
   U(w) = (w^(1-α) - 1) / (1 - α)  for α ≠ 1
   U(w) = log(w)                  for α = 1
   ```
   where `w` represents wealth, and `α` (the risk aversion parameter) determines an agent's sensitivity to risk.

2. **Prediction Markets**:
   The framework models a prediction market where agents trade based on their predictions. Each agent’s wealth is updated based on their prediction accuracy, and the equilibrium prices in the market reflect a weighted combination of the agents' beliefs.

3. **Bayesian Model Updates**:
   The wealth updates for isoelastic agents can simulate **Bayesian model updates**. When agents are rewarded based on their prediction performance, the market effectively updates the weights of different models or predictions in a principled way.

4. **Alpha-Mixtures**:
   The equilibrium prices in a market of isoelastic agents correspond to **α-mixtures**, where the weights of the mixture are directly influenced by the agents’ wealth.

5. **Performance**:
   The research demonstrates that inhomogeneous markets (markets with agents having diverse α values) outperform standard machine learning classifiers (e.g., random forests, neural networks, decision trees) on benchmark tasks.

### Contributions

- A novel iterative algorithm for computing market equilibrium.
- A demonstration of how wealth updates in prediction markets can act as a mechanism for model updates.
- Empirical evidence showing the superiority of the isoelastic agent-based approach over traditional classifiers.

---

## Repository Content

This repository provides a Python implementation of the algorithms and methodologies discussed in the paper using the PyTorch framework. Below is an outline of the key components of the codebase:

### 1. `isoelastic_agent.py`
This module implements the core functionality of isoelastic agents, including:
- The isoelastic utility function.
- Wealth update mechanisms for agents based on prediction market payoffs.
- Support for both logarithmic (α = 1) and general isoelastic (α ≠ 1) utility functions.

### 2. `market_simulation.py`
Contains code to simulate the prediction market:
- Initializes a market with multiple agents, each with their own utility function and wealth.
- Implements the iterative algorithm for computing market equilibrium.
- Updates agent wealth based on market payoffs.

### 3. `model_updates.py`
Demonstrates how wealth updates in the market can be translated into:
- Bayesian model updates.
- Mixture weight updates for combining predictions from multiple models.

### 4. `experiments/`
Contains scripts to reproduce the experiments from the paper:
- Benchmarks comparing the isoelastic market-based approach to standard classifiers like random forests and neural networks.
- Analysis of the impact of α (risk aversion parameter) on market performance.

### 5. `utils.py`
Utility functions for:
- Data preparation.
- Performance evaluation (e.g., accuracy, log-likelihood).
- Visualization of results (e.g., market equilibrium, wealth dynamics).

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch 1.10 or later
- Required Python packages: numpy, matplotlib, pandas

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/your-username/isoelastic-agents
   cd isoelastic-agents
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Running Market Simulations

To run a market simulation with isoelastic agents:
```
python market_simulation.py --config configs/market_config.json
```

Parameters such as the number of agents, the value of `α`, and the initial wealth distribution can be customized in the `configs/market_config.json` file.

### Reproducing Experiments

To reproduce the benchmark experiments from the paper:
```
python experiments/run_benchmarks.py
```

Results, including performance metrics and visualizations, will be saved in the `results/` directory.

---

## Example: Key Results

The following are some key findings from the paper, which can be reproduced using this implementation:

1. **Market Equilibrium**:
   The iterative algorithm successfully computes equilibrium prices, which reflect the α-mixture of agent beliefs.

2. **Classifier Performance**:
   Markets with isoelastic agents outperform traditional classifiers (e.g., random forests, neural networks) on benchmark datasets.

3. **Impact of α**:
   The diversity of risk aversion parameters (α values) among agents improves the robustness and accuracy of the market predictions.

---

## Citation

If you use this code or find the research helpful, please consider citing the original paper:

```
@article{storkey2012isoelastic,
  title={Isoelastic Agents and Wealth Updates in Machine Learning Markets},
  author={Storkey, Amos and Millin, Jono and Geras, Krzysztof},
  journal={arXiv preprint arXiv:1206.6443},
  year={2012}
}
```

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

This implementation is based on the paper **"Isoelastic Agents and Wealth Updates in Machine Learning Markets"** by Amos Storkey, Jono Millin, and Krzysztof Geras. Special thanks to the authors for their contributions to advancing prediction markets in machine learning.