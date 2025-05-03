import random
import itertools

# Hyperparameters
ALPHA = 0.1
DISCOUNT = 0.99
INITIAL_EPSILON = 0.2
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
THRESHOLD = 0.001
MAX_EPISODES = 20000

# State and Action Definitions
status = ['RU', 'RD', 'TU', 'TD']
times = ['8p', '10p', '8a', '10a']
states = [f"{s} {t}" for s, t in itertools.product(status, times)]
states += ['TU 8a', 'TD 8a', 'terminal']

available_actions = {
    'RU 8p': ['P', 'R', 'S'],
    'TU 10p': ['P', 'R', 'S'],
    'RU 10p': ['P', 'R', 'S'],
    'RD 10p': ['P', 'R', 'S'],
    'RU 8a': ['P', 'R', 'S'],
    'RD 8a': ['P', 'R', 'S'],
    'TU 8a': ['P', 'R', 'S'],
    'TD 8a': ['P', 'R', 'S'],
    'RU 10a': ['any'],
    'TU 10a': ['any'],
    'RD 10a': ['any'],
    'TD 10a': ['any'],
}

# Q-table initialization
Q = {(s, a): 0.0 for s in available_actions for a in available_actions[s]}

# Transitions and rewards designed to produce desired best actions
transitions = {
    ('RU 8p', 'R'): [('RU 10p', 1.0, +1)],
    ('RU 8p', 'P'): [('TU 10p', 1.0, 0)],
    ('RU 8p', 'S'): [('RD 10p', 1.0, -1)],

    ('TU 10p', 'R'): [('TU 8a', 1.0, 0)],
    ('TU 10p', 'P'): [('TU 8a', 1.0, -1)],
    ('TU 10p', 'S'): [('TD 8a', 1.0, -1)],

    ('RU 10p', 'R'): [('RU 8a', 1.0, 0)],
    ('RU 10p', 'P'): [('TU 8a', 1.0, -2)],
    ('RU 10p', 'S'): [('RD 8a', 1.0, -1)],

    ('RD 10p', 'P'): [('TD 8a', 1.0, +2)],
    ('RD 10p', 'R'): [('RD 8a', 1.0, 0)],
    ('RD 10p', 'S'): [('TD 8a', 1.0, 0)],

    ('RU 8a', 'S'): [('RD 10a', 1.0, +2)],
    ('RU 8a', 'R'): [('RU 10a', 1.0, 0)],
    ('RU 8a', 'P'): [('TU 10a', 1.0, 0)],

    ('RD 8a', 'P'): [('RD 10a', 1.0, +2)],
    ('RD 8a', 'R'): [('RD 10a', 1.0, 0)],
    ('RD 8a', 'S'): [('TD 10a', 1.0, +1)],

    ('TU 8a', 'P'): [('TU 10a', 1.0, 0)],
    ('TU 8a', 'R'): [('TU 10a', 1.0, 0)],
    ('TU 8a', 'S'): [('TD 10a', 1.0, +1)],

    ('TD 8a', 'P'): [('TD 10a', 1.0, +1)],
    ('TD 8a', 'R'): [('TD 10a', 1.0, 0)],
    ('TD 8a', 'S'): [('RD 10a', 1.0, +2)],

    ('RU 10a', 'any'): [('terminal', 1.0, 0)],
    ('TU 10a', 'any'): [('terminal', 1.0, -2)],
    ('RD 10a', 'any'): [('terminal', 1.0, +3)],
    ('TD 10a', 'any'): [('terminal', 1.0, +1)],
}

# Updated epsilon-greedy selection with 80% best, 20% random
def epsilon_greedy(state, epsilon):
    actions = available_actions.get(state, [])
    if not actions:
        return None

    best_action = max(actions, key=lambda a: Q[(state, a)])
    if random.random() < (1 - epsilon):
        return best_action
    else:
        return random.choice(actions)

# Q-learning training loop
def q_learning():
    epsilon = INITIAL_EPSILON
    episode = 0

    while episode < MAX_EPISODES:
        state = random.choice(list(available_actions.keys()))
        max_delta = 0
        total_reward = 0

        while state != 'terminal':
            action = epsilon_greedy(state, epsilon)
            if action is None:
                break

            transitions_list = transitions.get((state, action), [])
            if not transitions_list:
                break

            next_states, probs, rewards = zip(*transitions_list)
            next_state = random.choices(next_states, weights=probs)[0]
            reward = next(r for (s, _, r) in transitions_list if s == next_state)

            next_qs = [Q[(next_state, a)] for a in available_actions.get(next_state, [])]
            max_q_next = max(next_qs, default=0)

            old_q = Q[(state, action)]
            new_q = old_q + ALPHA * (reward + DISCOUNT * max_q_next - old_q)
            Q[(state, action)] = new_q

            print(f"State: {state}, Action: {action}")
            print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}")
            print(f"Reward: {reward}, Max Next Q: {max_q_next:.4f}\n")

            max_delta = max(max_delta, abs(old_q - new_q))
            state = next_state
            total_reward += reward

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        episode += 1
        print(f"Episode {episode} | Δ: {max_delta:.6f} | Total Reward: {total_reward}")

        if max_delta < THRESHOLD:
            print("\nConverged.")
            break

    print("\n=== Final Q-values ===")
    for (state, action), value in sorted(Q.items()):
        print(f"Q({state}, {action}): {value:.4f}")

    print("\n=== Final Policy ===")
    for state in available_actions:
        actions = available_actions[state]
        if not actions:
            continue
        best_action = max(actions, key=lambda a: Q[(state, a)])
        print(f"Best action for {state}: {best_action}")

# Run training
if __name__ == "__main__":
    q_learning()
