import numpy as np

# methods
## method that calculate the result
def pull_arm(prob, rng):

    random_value = rng.random()
    reward = 0

    if random_value < prob:
        reward = reward + 1

    return reward


## calculate the average probability for each arm after iterating
def get_probs(reward, times, esti_probs):

    for i in range(len(times)):
        if times[i] == 0:
            esti_probs[i] = np.nan
            print("arm ", i , "was never selected; its average reward is undefined.")
        else:
            esti_probs[i] = reward[i] / times[i]

    return esti_probs


## make multi times decition
def run_arms(reward, times, num_dec, true_probs, rng):

    for i in range(num_dec):
        action = choose_arm(len(true_probs), rng)
        reward[action] = reward[action] + pull_arm(true_probs[action], rng)
        times[action] = times[action] + 1

    return reward, times


## method that randomly choose one arm
def choose_arm(num_arms, rng):

    action = rng.integers(num_arms)

    return action





## method that choose the arm
def choose_arm_EG(epsilon, esti_probs_EG, rng):
    num = np.argmax(esti_probs_EG)
    random = rng.random()
    if random > epsilon:
        action = num
    else:
        action = rng.integers(len(esti_probs_EG))
    return action


## method that use EG
def EG(epsilon, reward_EG, times_EG, num_dec, esti_probs_EG, true_probs, rng):
    for i in range(len(times_EG)):
        if times_EG[i] == 0:
            action = i
            reward_EG[action] = reward_EG[action] + pull_arm(true_probs[action], rng)
            times_EG[action] = times_EG[action] + 1
            esti_probs_EG[action] = reward_EG[action] / times_EG[action]

    for i in range(num_dec - len(true_probs)):
        action = choose_arm_EG(epsilon, esti_probs_EG, rng)
        reward_EG[action] = reward_EG[action] + pull_arm(true_probs[action], rng)
        times_EG[action] = times_EG[action] + 1
        esti_probs_EG[action] = reward_EG[action] / times_EG[action]

    return reward_EG, times_EG, esti_probs_EG



# setup
rng = np.random.default_rng(17)
true_probs = np.array([0.3, 0.5, 0.7])
reward = np.zeros(len(true_probs))
times = np.zeros(len(true_probs))
esti_probs = np.zeros(len(true_probs))

num_dec = 10000


reward, times = run_arms(reward, times, num_dec, true_probs, rng)
esti_probs = get_probs(reward, times, esti_probs)


print("chosen arm: ", times)
print("true probability: ", true_probs)
print("reward: ", reward)
print("estimate probability: ", esti_probs)

## epsilon-greedy set up
reward_EG = np.zeros(len(true_probs))
times_EG = np.zeros(len(true_probs))
esti_probs_EG = np.zeros(len(true_probs))


epsilon = 0.1


reward_EG, times_EG, esti_probs_EG = EG(epsilon, reward_EG, times_EG, num_dec, esti_probs_EG, true_probs, rng)


print("chosen arm EG: ", times_EG)
print("true probability EG: ", true_probs)
print("reward EG: ", reward_EG)
print("estimate probability EG: ", esti_probs_EG)