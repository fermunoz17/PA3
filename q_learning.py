import random

# Hyperparameters
ALPHA = 0.1        # Learning rate
DISCOUNT = 0.99    # Discount factor (lambda)
EPSILON = 0.2      # 20% random exploration
THRESHOLD = 0.001  # Convergence threshold
MAX_EPISODES = 10000

# States and Actions
states = ['R', 'T', '8p']  # '8p' = terminal state
actions = ['Party', 'Rest', 'Study']

# Available actions per state
available_actions = {
    'R': ['Party', 'Study', 'Rest'],
    'T': ['Party', 'Study', 'Rest'],
    '8p': []  # No actions at terminal
}

# Transitions: (state, action) -> list of (next_state, probability, reward)
# Adding small chance to transition to '8p' (terminal)
transitions = {
    ('R', 'Party'): [('T', 0.9, -5), ('8p', 0.1, 0)],
    ('R', 'Study'): [('R', 0.9, 10), ('8p', 0.1, 0)],
    ('R', 'Rest'):  [('R', 0.9, 1), ('8p', 0.1, 0)],
    ('T', 'Party'): [('T', 0.9, -10), ('8p', 0.1, 0)],
    ('T', 'Study'): [('T', 0.9, 2), ('8p', 0.1, 0)],
    ('T', 'Rest'):  [('R', 0.9, 2), ('8p', 0.1, 0)],
}

# Initialize Q-values only for valid (state, action) pairs
Q = {(s, a): 0.0 for s in states for a in available_actions[s]}

def epsilon_greedy(state):
    actions_here = available_actions[state]
    if not actions_here:
        return None
    if random.random() < EPSILON:
        return random.choice(actions_here)
    else:
        q_vals = {a: Q[(state, a)] for a in actions_here}
        return max(q_vals, key=q_vals.get)

def q_learning():
    episode = 0
    converged = False

    while episode < MAX_EPISODES:
        max_delta = 0
        current_state = random.choice(['R', 'T'])

        while current_state != '8p':
            action = epsilon_greedy(current_state)
            if not action:
                break

            possible_transitions = transitions.get((current_state, action), [])
            if not possible_transitions:
                break

            next_states, probs, rewards = zip(*possible_transitions)
            next_state = random.choices(next_states, weights=probs)[0]
            reward = [r for (s, p, r) in possible_transitions if s == next_state][0]

            max_q_next = max([Q.get((next_state, a), 0) for a in available_actions.get(next_state, [])], default=0)
            old_q = Q[(current_state, action)]
            new_q = old_q + ALPHA * (reward + DISCOUNT * max_q_next - old_q)
            Q[(current_state, action)] = new_q

            # Logging
            print(f"State: {current_state}, Action: {action}")
            print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}")
            print(f"Reward: {reward}, Max Next Q: {max_q_next:.4f}\n")

            max_delta = max(max_delta, abs(old_q - new_q))
            current_state = next_state

        episode += 1
        print(f"Episode {episode} | Max Delta: {max_delta:.6f}")

        # Only break if global change is small
        if max_delta < THRESHOLD:
            print("\n✅ Converged!")
            converged = True
            break

    if not converged:
        print("\n❗ Stopped after reaching maximum episodes without convergence.")

    # Final Output
    print(f"\n=== Episodes: {episode} ===")
    print("\n=== Final Q-Values ===")
    for (state, action), value in Q.items():
        print(f"Q({state}, {action}): {value:.4f}")

    print("\n=== Final Policy ===")
    for state in ['R', 'T']:
        q_values = {a: Q[(state, a)] for a in available_actions[state]}
        best_action = max(q_values, key=q_values.get)
        print(f"Best action for {state}: {best_action}")


if __name__ == "__main__":
    q_learning()
