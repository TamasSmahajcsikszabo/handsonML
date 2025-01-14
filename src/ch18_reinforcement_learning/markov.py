'''
    Andrey Markov:
        - stochastic processes with no memory (Markov chains)
        - probability to evolve between s and s' is fixed and is conditional
        only on s and s' pair, not on past states
        - terminal state (node with 100%)
        - Richard Bellman: Markov decision processes
            - at each step an agent can choose
            - choice have an impact on transition probability
            - introduces rewards (positive, negative)
            - the agent's goal is to maximize the reward over time
            - optimal state value of any state s is V*(s)
            V*(s) = sum of all duscoiunted future rewards the agent can
            expect on average after it reaches the state, assuming it acts
            optimally
'''
