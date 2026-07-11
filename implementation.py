import numpy as np
import torch
import torch.nn.functional as F

class IsoelasticAgent:
    def __init__(self, wealth, risk_aversion):
        """
        Initialize an isoelastic agent.
        :param wealth: Initial wealth of the agent.
        :param risk_aversion: Risk aversion parameter (rho). rho > 0 for isoelastic utility.
        """
        self.wealth = wealth
        self.risk_aversion = risk_aversion

    def utility(self, payoff):
        """
        Compute the utility of a given payoff.
        :param payoff: Payoff received by the agent.
        :return: Utility value.
        """
        if self.risk_aversion == 1:
            # Logarithmic utility
            return torch.log(self.wealth + payoff)
        else:
            # Isoelastic utility
            return ((self.wealth + payoff) ** (1 - self.risk_aversion)) / (1 - self.risk_aversion)

    def update_wealth(self, payoff):
        """
        Update the agent's wealth based on the payoff.
        :param payoff: Payoff received by the agent.
        """
        self.wealth += payoff


class PredictionMarket:
    def __init__(self, agents, alpha=0.5):
        """
        Initialize a prediction market with isoelastic agents.
        :param agents: List of IsoelasticAgent objects.
        :param alpha: Mixing parameter for alpha-mixtures.
        """
        self.agents = agents
        self.alpha = alpha

    def equilibrium_price(self, predictions):
        """
        Compute the equilibrium price as an alpha-mixture of agent predictions weighted by wealth.
        :param predictions: List of predictions from agents.
        :return: Equilibrium price.
        """
        weights = torch.tensor([agent.wealth for agent in self.agents], dtype=torch.float32)
        weights /= weights.sum()  # Normalize weights by total wealth
        mixture = torch.tensor(predictions, dtype=torch.float32)
        equilibrium = torch.sum(weights * mixture ** self.alpha)
        return equilibrium

    def update_agents(self, true_value):
        """
        Update agents' wealth based on their prediction and the true value.
        :param true_value: The true value of the prediction target.
        """
        for agent in self.agents:
            payoff = -torch.abs(agent.prediction - true_value)  # Negative error as payoff
            agent.update_wealth(payoff.item())


if __name__ == '__main__':
    # Dummy data for demonstration
    num_agents = 5
    initial_wealth = 100.0
    risk_aversion_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    predictions = [0.2, 0.4, 0.6, 0.8, 0.5]
    true_value = 0.7

    # Initialize agents
    agents = [IsoelasticAgent(initial_wealth, rho) for rho in risk_aversion_values]
    for agent, pred in zip(agents, predictions):
        agent.prediction = pred

    # Initialize prediction market
    market = PredictionMarket(agents)

    # Compute equilibrium price
    equilibrium_price = market.equilibrium_price(predictions)
    print(f"Equilibrium Price: {equilibrium_price.item()}")

    # Update agents based on true value
    market.update_agents(true_value)

    # Print updated wealth of agents
    for i, agent in enumerate(agents):
        print(f"Agent {i+1} Wealth: {agent.wealth}")