import numpy as np


def utility(v, t, u, reward, gamma):
    """Return the state utility.

    @param v the state vector
    @param t transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_utility = np.zeros(4)
    for action in range(0, 4):
        action_utility[action] = np.sum(np.multiply(u, np.dot(v, t[:, :, action])))
    future_utility = max(action_utility)
    return reward + gamma * future_utility


def value_iteration(t, u_init, r, gamma, epsilon):
    u = u_init.copy()

    states_count = r.shape[0]

    iteration = 0

    while True:
        delta = 0
        u0 = u.copy()
        iteration += 1
        for s in range(states_count):
            reward = r[s]
            v = np.zeros((1, states_count))
            v[0, s] = 1.0
            u[s] = utility(v, t, u0, reward, gamma)
            delta = max(delta, np.abs(u[s] - u0[s]))

        if delta < epsilon * (1 - gamma) / gamma:
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            print(u[0:4])
            print(u[4:8])
            print(u[8:12])
            print("===================================================")
            break


def policy_evaluation(p, u, r, t, gamma):
    """
    Return the policy utility.
    @param p policy vector
    @param u utility vector
    @param r reward vector
    @param t transition matrix
    @param gamma discount factor
    @return the utility vector u
    """
    for s in range(12):
        if np.isnan(p[s]):
            continue
        v = np.zeros((1, 12))
        v[0, s] = 1.0
        action = int(p[s])
        u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, t[:, :, action])))

    return u


def expected_action(u, t, v):
    """Return the expected action.

    It returns an action based on the
    expected utility of doing a in state s,
    according to t and u. This action is
    the one that maximize the expected
    utility.
    @param u utility vector
    @param t transition matrix
    @param v starting vector
    @return expected action (int)
    """
    actions_utility = np.zeros(4)
    for action in range(4):
        actions_utility[action] = np.sum(np.multiply(u, np.dot(v, t[:, :, action])))

    return np.argmax(actions_utility)


def print_policy(p, shape):
    """Printing utility.

    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    # obstacles
    """
    state = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if p[state] == -1:
                policy_string += " *  "
            elif p[state] == 0:
                policy_string += " ^  "
            elif p[state] == 1:
                policy_string += " <  "
            elif p[state] == 2:
                policy_string += " v  "
            elif p[state] == 3:
                policy_string += " >  "
            elif np.isnan(p[state]):
                policy_string += " #  "
            state += 1
        policy_string += '\n'
    print(policy_string)


def policy_iteration(t, u_init, r, gamma, epsilon):
    u = u_init.copy()

    p = np.random.randint(0, 4, size=12).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    iteration = 0

    while True:
        iteration += 1
        # 1- Policy evaluation
        u0 = u.copy()
        u = policy_evaluation(p, u, r, t, gamma)
        delta = np.absolute(u - u0).max()

        if delta < epsilon * (1 - gamma) / gamma:
            break

        for s in range(12):
            if not np.isnan(p[s]) and p[s] != -1:
                v = np.zeros((1, 12))
                v[0, s] = 1.0
                # 2- Policy improvement
                p[s] = expected_action(u, t, v)

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3, 4))
    print("===================================================")


def run():
    # Starting state vector
    # The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0]])

    # Utility vector
    u_optimal = np.array([[0.812, 0.868, 0.918, 1.0,
                           0.762, 0.0, 0.660, -1.0,
                           0.705, 0.655, 0.611, 0.388]])

    # Transition matrix loaded from file
    t = np.load("t.npy")

    u_init = np.array([0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0])

    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    # Defining the reward for state (1,1)
    reward = r[8]

    # value iteration parameters
    gamma = 0.9999
    epsilon = 0.0001

    # Use the Bellman equation to find the utility of state (1,1)
    utility_11 = utility(v, t, u_optimal, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))

    print("running value iteration")
    value_iteration(t, u_init, r, gamma, epsilon)

    print("running policy iteration")
    policy_iteration(t, u_init, r, gamma, epsilon)


run()
