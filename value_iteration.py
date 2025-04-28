# Discount factor
DISCOUNT = 0.99

# Convergence threshold
THRESHOLD = 0.001

# MDP
states = ['RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a', 'TU8a', 'TU10a', 'RU10a', 'RD10a', 'TD10a', 'terminal']
actions = ['Party', 'Rest', 'Study', 'any']

# Define available actions per state
available_actions = {
    'RU8p': ['Party', 'Rest', 'Study'],
    'TU10p': ['Party', 'Rest', 'Study'],
    'RD10p': ['Party', 'Rest'],
    'RU10p': ['Party', 'Rest', 'Study'],
    'RU8a': ['Party', 'Rest', 'Study'],
    'RD8a': ['Party', 'Rest'],
    'TU10a': ['any'],
    'RU10a': ['any'],
    'RD10a': ['any'],
    'TD10a': ['any'],
    'terminal': []  # terminal state, no actions
}


# Transition probabilities and rewards
# Format: (next_state, probability, reward)
transitions = {
    # From RU8p
    ('RU8p', 'Party'): [('TU10p', 1.0, +2)],
    ('RU8p', 'Rest'): [('RU10p', 1.0, 0)],
    ('RU8p', 'Study'): [('RD10p', 1.0, -1)],

    # From TU10p
    ('TU10p', 'Party'): [('TU8a', 1.0, +2)],
    ('TU10p', 'Rest'): [('RU8a', 1.0, 0)],
    ('TU10p', 'Study'): [('RU8a', 1.0, 0)],  # same as Rest, it looks like

    # From RD10p
    ('RD10p', 'Party'): [('RD8a', 1.0, +2)],
    ('RD10p', 'Rest'): [('RD8a', 1.0, 0)],

    # From RU10p
    ('RU10p', 'Party'): [('RU8a', 1.0, +2)],
    ('RU10p', 'Rest'): [('RU8a', 0.5, 0), ('RD8a', 0.5, +5)],
    ('RU10p', 'Study'): [('RD8a', 1.0, -1)],

    # From RU8a
    ('RU8a', 'Party'): [('TU10a', 1.0, +2)],
    ('RU8a', 'Rest'): [('RU10a', 1.0, 0)],
    ('RU8a', 'Study'): [('RD10a', 1.0, -1)],

    # From RD8a
    ('RD8a', 'Party'): [('RD10a', 1.0, +2)],
    ('RD8a', 'Rest'): [('RD10a', 1.0, 0)],

    # From TU10a
    ('TU10a', 'any'): [('terminal', 1.0, -1)],

    # From RU10a
    ('RU10a', 'any'): [('terminal', 1.0, 0)],

    # From RD10a
    ('RD10a', 'any'): [('terminal', 1.0, +4)],

    # From TD10a
    ('TD10a', 'any'): [('terminal', 1.0, +3)],
}


# Initialize state values
V = {state: 0.0 for state in states}

def value_iteration():
    iteration = 0
    while True:
        delta = 0
        print(f"\nIteration {iteration}")
        for state in states:
            if state == '8p':
                continue  # terminal state
            action_values = {}
            for action in available_actions.get(state, []):
                value = 0
                for (next_state, prob, reward) in transitions.get((state, action), []):
                    value += prob * (reward + DISCOUNT * V[next_state])
                action_values[action] = value
            
            if action_values:
                max_action = max(action_values, key=action_values.get)
                max_value = action_values[max_action]
                old_value = V[state]
                V[state] = max_value

                print(f"State: {state}")
                print(f"Old Value: {old_value:.4f}, New Value: {max_value:.4f}")
                print(f"Action Values: {action_values}")
                print(f"Selected Action: {max_action}\n")

                delta = max(delta, abs(old_value - max_value))

        iteration += 1
        if delta < THRESHOLD:
            break

    print("\n=== Final Values ===")
    for state in states:
        print(f"{state}: {V[state]:.4f}")

    print("\n=== Final Policy ===")
    for state in states:
        if state == '8p':
            continue
        action_values = {}
        for action in available_actions.get(state, []):
            value = 0
            for (next_state, prob, reward) in transitions.get((state, action), []):
                value += prob * (reward + DISCOUNT * V[next_state])
            action_values[action] = value
        if action_values:
            best_action = max(action_values, key=action_values.get)
            print(f"Best action for {state}: {best_action}")

if __name__ == "__main__":
    value_iteration()
