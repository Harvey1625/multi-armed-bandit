import numpy as np
import matplotlib.pyplot as plt

# methods

## method that simulates one Bernoulli reward

def pull_arm(prob, rng):

    random_value = rng.random()
    reward = 0

    if random_value < prob:
        reward = reward + 1

    return reward


## method that calculates the estimated probabilities

def get_probs(reward, times, esti_probs):

    for i in range(len(times)):
        if times[i] == 0:
            esti_probs[i] = np.nan
            print("arm", i + 1, "was never selected; its average reward is undefined.")

        else:
            esti_probs[i] = reward[i] / times[i]

    return esti_probs


## method that randomly chooses one arm

def choose_arm(num_arms, rng):

    action = rng.integers(num_arms)

    return action


## method that runs the random

def run_random(num_dec, true_probs, rng):

    reward = np.zeros(len(true_probs), dtype=int)
    times = np.zeros(len(true_probs), dtype=int)
    esti_probs = np.zeros(len(true_probs))
    reward_history = np.zeros(num_dec, dtype=int)
    action_history = np.zeros(num_dec, dtype=int)

    for i in range(num_dec):

        action = choose_arm(len(true_probs), rng)
        current_reward = pull_arm(true_probs[action], rng)
        reward[action] = reward[action] + current_reward
        times[action] = times[action] + 1
        reward_history[i] = current_reward
        action_history[i] = action

    esti_probs = get_probs(reward, times, esti_probs)

    return reward, times, esti_probs, reward_history, action_history


## method that chooses one arm using epsilon-greedy

def choose_arm_EG(epsilon, esti_probs_EG, rng):

    num = np.argmax(esti_probs_EG)
    random_value = rng.random()

    if random_value > epsilon:
        action = num

    else:
        action = rng.integers(len(esti_probs_EG))

    return action


## method that runs the epsilon-greedy
def EG(epsilon, num_dec, true_probs, rng):

    reward_EG = np.zeros(len(true_probs), dtype=int)
    times_EG = np.zeros(len(true_probs), dtype=int)
    esti_probs_EG = np.zeros(len(true_probs))
    reward_history_EG = np.zeros(num_dec, dtype=int)
    action_history_EG = np.zeros(num_dec, dtype=int)

    ## select each arm once for initialization

    for i in range(len(true_probs)):
        action = i
        current_reward = pull_arm(true_probs[action], rng)
        reward_EG[action] = reward_EG[action] + current_reward
        times_EG[action] = times_EG[action] + 1
        esti_probs_EG[action] = reward_EG[action] / times_EG[action]
        reward_history_EG[i] = current_reward
        action_history_EG[i] = action

    ## run epsilon-greedy for the remaining decisions

    for i in range(len(true_probs), num_dec):
        action = choose_arm_EG(epsilon, esti_probs_EG, rng)
        current_reward = pull_arm(true_probs[action], rng)
        reward_EG[action] = reward_EG[action] + current_reward
        times_EG[action] = times_EG[action] + 1
        esti_probs_EG[action] = reward_EG[action] / times_EG[action]
        reward_history_EG[i] = current_reward
        action_history_EG[i] = action

    return reward_EG, times_EG, esti_probs_EG, reward_history_EG, action_history_EG


# setup

rng = np.random.default_rng(17)
true_probs = np.array([0.3, 0.5, 0.7])
num_dec = 10000


## random setup

reward, times, esti_probs, reward_history, action_history = run_random(num_dec, true_probs, rng)


print("arm pull counts: ", times)
print("true probabilities: ", true_probs)
print("total rewards: ", reward)
print("estimated probabilities: ", esti_probs)


## epsilon-greedy setup

epsilon = 0.1

reward_EG, times_EG, esti_probs_EG, reward_history_EG, action_history_EG = EG(epsilon, num_dec, true_probs, rng)

## calculate cumulative average rewards

cumu_avg_rand = np.cumsum(reward_history) / np.arange(1, num_dec + 1)
cumu_avg_EG = np.cumsum(reward_history_EG) / np.arange(1, num_dec + 1)


## calculate optimal arm selection rates

opti_arm = np.argmax(true_probs)
opti_rate_rand = np.mean(action_history == opti_arm)
opti_rate_EG = np.mean(action_history_EG == opti_arm)


print("arm pull counts EG: ", times_EG)
print("true probabilities EG: ", true_probs)
print("total rewards EG: ", reward_EG)
print("estimated probabilities EG: ", esti_probs_EG)
print("overall average reward random: ", cumu_avg_rand[-1])
print("overall average reward EG: ", cumu_avg_EG[-1])
print("optimal arm selection rate random: ", opti_rate_rand)
print("optimal arm selection rate EG: ", opti_rate_EG)


## plot cumulative average rewards

dec_steps = np.arange(1, num_dec + 1)

plt.figure(figsize=(10, 6))
plt.plot(dec_steps, cumu_avg_rand, label="Random")
plt.plot(dec_steps, cumu_avg_EG, label="Epsilon-Greedy")
plt.axhline(y=np.max(true_probs), color="black", linestyle="--", label="Optimal Expected Reward")

plt.xlabel("Decision")
plt.ylabel("Cumulative Average Reward")
plt.title("Random vs. Epsilon-Greedy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("reward_comparison.png", dpi=300)
plt.show()
