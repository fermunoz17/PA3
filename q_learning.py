import random
import itertools

# Hyperparameters
ALPHA = 0.1
DISCOUNT = 0.99
INITIAL_EPSILON = 0.2
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
THRESHOLD = 0.0001
MAX_EPISODES = 10000

# Define all composite states from the MDP
status = ['RU', 'RD', 'TU', 'TD']
times = ['8p', '10p', '8a', '10a']
states = [f"{s} {t}" for s, t in itertools.product(status, times)]
states += ['TU 8a', 'TD 8a', 'terminal']

# Available actions per state
available_actions = {
    'RU 8p': ['P', 'R', 'S'],
    'TU 10p': ['P', 'R', 'S'],
    'RU 10p': ['P', 'R', 'S'],
    'RD 10p': ['P', 'R', 'S'],
    'RU 8a': ['P', 'R', 'S'],
    'RD 8a': ['P', 'R', 'S'],
    'RU 10a': ['any'],
    'TU 10a': ['any'],
    'RD 10a': ['any'],
    'TD 10a': ['any'],
    'TU 8a': ['P', 'R', 'S'],
    'TD 8a': ['P', 'R', 'S']
}

# Initialize Q-values
Q = {(s, a): 0.0 for s in available_actions for a in available_actions[s]}

# Transitions from the diagram
transitions = {
    ('RU 8p', 'P'): [('TU 10p', 1.0, +2)],
    ('RU 8p', 'R'): [('RU 10p', 1.0, 0)],
    ('RU 8p', 'S'): [('RD 10p', 1.0, -1)],
    ('TU 10p', 'P'): [('TU 8a', 1.0, +2)],
    ('RU 10p', 'R'): [('RU 8a', 1.0, 0)],
    ('RU 10p', 'P'): [('RU 8a', 0.5, +2), ('TU 8a', 0.5, -2)],
    ('RU 10p', 'S'): [('RD 8a', 1.0, -1)],
    ('RD 10p', 'R'): [('RD 8a', 1.0, 0)],
    ('RD 10p', 'P'): [('RD 8a', 0.5, +2), ('TD 8a', 0.5, +2)],
    ('RD 10p', 'S'): [('TD 8a', 1.0, +2)],
    ('RU 8a', 'P'): [('TU 10a', 1.0, +2)],
    ('RU 8a', 'R'): [('RU 10a', 1.0, 0)],
    ('RU 8a', 'S'): [('RD 10a', 1.0, -1)],
    ('RD 8a', 'P'): [('RD 10a', 1.0, +2)],
    ('RD 8a', 'R'): [('RD 10a', 1.0, 0)],
    ('RD 8a', 'S'): [('TD 10a', 1.0, +2)],
    ('TU 10a', 'any'): [('terminal', 1.0, -1)],
    ('RU 10a', 'any'): [('terminal', 1.0, 0)],
    ('RD 10a', 'any'): [('terminal', 1.0, +4)],
    ('TD 10a', 'any'): [('terminal', 1.0, +3)],
}

# Epsilon-greedy action selection
def epsilon_greedy(state, epsilon):
    actions_here = available_actions[state]
    if not actions_here:
        return None
    if random.random() < epsilon:
        return random.choice(actions_here)
    q_vals = {a: Q[(state, a)] for a in actions_here}
    return max(q_vals, key=q_vals.get)

# Main Q-learning loop
def q_learning():
    episode = 0
    epsilon = INITIAL_EPSILON
    converged = False

    while episode < MAX_EPISODES:
        max_delta = 0
        total_reward = 0
        current_state = random.choice(list(available_actions.keys()))

        while current_state != 'terminal':
            action = epsilon_greedy(current_state, epsilon)
            if not action:
                break

            transitions_list = transitions.get((current_state, action), [])
            if not transitions_list:
                break

            next_states, probs, rewards = zip(*transitions_list)
            next_state = random.choices(next_states, weights=probs)[0]
            reward = [r for (s, p, r) in transitions_list if s == next_state][0]
            total_reward += reward

            max_q_next = max(
                [Q.get((next_state, a), 0) for a in available_actions.get(next_state, [])],
                default=0
            )
            old_q = Q[(current_state, action)]
            new_q = old_q + ALPHA * (reward + DISCOUNT * max_q_next - old_q)
            Q[(current_state, action)] = new_q

            # ✅ Required printout per assignment
            print(f"State: {current_state}, Action: {action}")
            print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}")
            print(f"Reward: {reward}, Max Next Q: {max_q_next:.4f}\n")

            max_delta = max(max_delta, abs(old_q - new_q))
            current_state = next_state

        episode += 1
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        print(f"Episode {episode} | Max Δ: {max_delta:.6f} | Total Reward: {total_reward}")

        if max_delta < THRESHOLD:
            print("\nConverged!")
            converged = True
            break

    print(f"\n=== Final Q-Values ===")
    for (state, action), value in sorted(Q.items()):
        print(f"Q({state}, {action}): {value:.4f}")

    print(f"\n=== Final Policy ===")
    for state in available_actions:
        actions_here = available_actions[state]
        if not actions_here:
            continue
        best_action = max(actions_here, key=lambda a: Q[(state, a)])
        print(f"Best action for {state}: {best_action}")

# Run the learning process
if __name__ == "__main__":
    q_learning()
