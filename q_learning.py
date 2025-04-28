import random

# Learning rate, discount factor, epsilon
ALPHA = 0.1
DISCOUNT = 0.99
EPSILON = 0.2
THRESHOLD = 0.001

states = ['R', 'T', 'D', 'U', '8p']
actions = ['P', 'R', 'S']

available_actions = {
    'R': ['Party', 'Study', 'Rest'],
    'T': ['Party', 'Study', 'Rest'],
    'D': ['any'],
    'U': ['any'],
    '8p': []
}

transitions = {
    ('R', 'Party'): [('T', 1.0, -5)],
    ('R', 'Study'): [('R', 1.0, 10)],
    ('T', 'Rest'): [('R', 1.0, 2)],
    ('T', 'Party'): [('T', 1.0, -10)],
    # add more transitions here
}

Q = {(state, action): 0.0 for state in states for action in actions}

def epsilon_greedy(state):
    if random.random() < EPSILON:
        return random.choice(available_actions.get(state, []))
    else:
        q_values = {action: Q.get((state, action), 0) for action in available_actions.get(state, [])}
        if q_values:
            return max(q_values, key=q_values.get)
        else:
            return None

def q_learning():
    episode = 0
    while True:
        delta = 0
        current_state = random.choice(states[:-1])  # pick a non-terminal state to start
        while current_state != '8p':
            action = epsilon_greedy(current_state)
            if not action:
                break
            transitions_list = transitions.get((current_state, action), [])
            if not transitions_list:
                break
            next_state, prob, reward = random.choice(transitions_list)

            old_q = Q[(current_state, action)]
            max_q_next = max([Q.get((next_state, a), 0) for a in available_actions.get(next_state, [])], default=0)
            new_q = old_q + ALPHA * (reward + DISCOUNT * max_q_next - old_q)
            Q[(current_state, action)] = new_q

            print(f"State: {current_state}, Action: {action}")
            print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}")
            print(f"Reward: {reward}, Max Next Q: {max_q_next:.4f}\n")

            delta = max(delta, abs(old_q - new_q))

            current_state = next_state

        episode += 1
        if delta < THRESHOLD:
            break

    print("\n=== Final Q-Values ===")
    for (state, action), value in Q.items():
        if action in available_actions.get(state, []):
            print(f"Q({state}, {action}): {value:.4f}")

    print("\n=== Final Policy ===")
    for state in states:
        if state == '8p':
            continue
        q_values = {action: Q.get((state, action), 0) for action in available_actions.get(state, [])}
        if q_values:
            best_action = max(q_values, key=q_values.get)
            print(f"Best action for {state}: {best_action}")

if __name__ == "__main__":
    q_learning()
