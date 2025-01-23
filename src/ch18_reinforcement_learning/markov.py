'''
    PG algorithms tried to directly optimize the policy to increase reweards
    In the following family of algorithms, the agent learns to estimate the
    expected return fo-r each state and each action within them to decide
    next action

    MDP
    Andrey Markov:
        - fixed number of states
        - the process randomly evolves from one state to the next
        - stochastic processes with no memory (Markov chains)
        - probability to evolve between s and s' is fixed and is conditional
        only on s and s' pair, not on past states
        - terminal state (node with 100% stays there and cannot exit)
        - Richard Bellman (1950s): Markov decision processes
            - at each step an agent can choose
            - choice have an impact on transition probability
            - introduces rewards (positive, negative)
            - the agent's goal is to maximize the reward over time
            - **optimal state value of any state** s is V*(s)
            V*(s) = sum of all discounted future rewards the agent can
            expect on average after it reaches the state, assuming it acts
            optimally
            - Bellman optimality equation:
                V*(s) = max Sum T(s, a, s')[R(s,a,s') + gamma * V*(s')] for
                all s states
                - T(s,a,s') is the transition probability from s to s' given the agent
                has chosen action a
                - R(s,a,s') is the reward the agent gets when it geoes from
                state s to s' chosing action a
                - gamma is the discount factor
            - using *value iteration algorithm (VIA) * it can compute the optimal state value
            estimates for all states
            - over time it will converge to the optimal estimates
            - for the k-th iteration:
            V_k+1 (s) <- max Sum T(s, a, s')[R(s,a,s') + gamma * V_k(s')]
            V_k(s) is the estimated value of state s in the k-th iteration
            of the algorithm
            -estimating the optimal state-action values Q-values (quality values)
            - the optimal Q-valuye of the state-action pair (s, a) Q*(s,a)
            is the sum of all discounted future rewards the agent can expect
            on average after reaching s state and choosing action a, but before
            it sees the outcome of the action
            - all Q-value estimates start as zero
            - iteratively updating them using the Q-value iteration algorithm
            Q_k+1 (s) <- sum T(s, a, s')[R(s,a,s') + gamma * max Q_k(a', s')]
            for all (s, a)

'''
from typing import NewType, List
import numpy as np

State = NewType('State', float)
Action = NewType('Action', float)
Transition = List[State | Action] | None

transition_probabilities: List[List[Transition]] = [
[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]
]

rewards = [
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0,0,0], [0,0,0], [0,0,-50]],
    [[0,0,],[40,0,0],[0,0,0]]
]
possible_actions = [
    [0, 1, 2],
    [0, 2],
    [1]
]

Q_values = np.full((3,3), -np.inf)

for state, actions in enumerate(possible_actions):
    Q_values[state,  actions] = 0.0

# let's run the Q-value iteration algorithm
gamma = 0.9

for iteration in range(50):
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]:
            print(f"possible action {a}")
            print(f"updating: {Q_values[s,a]}")
            Q_values[s, a] = np.sum([
                transition_probabilities[s][a][sp] * (rewards[s][a][sp] + gamma * Q_prev[sp].max()) for sp in range(3)
            ])


# optimal action for each state:
Q_values.argmax(axis=1)
