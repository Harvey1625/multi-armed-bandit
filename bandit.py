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


## method that selects an arm using the random policy

def policy_random(times, reward, esti_probs, rng, i, epsilon):

    action = rng.integers(len(times))

    return action


## method that selects an arm using the epsilon-greedy policy

def policy_EG(times, reward, esti_probs, rng, i, epsilon):

    if i < len(times):
        action = i

    else:
        random_value = rng.random()

        if random_value > epsilon:
            action = np.argmax(esti_probs)

        else:
            action = rng.integers(len(times))

    return action


## method that selects an arm using the UCB policy

def policy_UCB(times, reward, esti_probs, rng, i, epsilon):

    if i < len(times):
        action = i

    else:
        bonus = np.sqrt(2 * np.log(i + 1) / times)
        ucb_values = esti_probs + bonus
        action = np.argmax(ucb_values)

    return action


## method that selects an arm using the TS policy

def policy_TS(times, reward, esti_probs, rng, i, epsilon):

    alpha = reward + 1
    beta = times - reward + 1

    samples = rng.beta(alpha, beta)

    action = np.argmax(samples)

    return action


## method that runs a bandit simulation using a given policy

def run_bandit(num_dec, true_probs, policy, rng, epsilon=0.1, reward_rng=None):

    num_arms = len(true_probs)

    if reward_rng is None:
        reward_rng = rng

    reward = np.zeros(num_arms, dtype=int)
    times = np.zeros(num_arms, dtype=int)
    esti_probs = np.zeros(num_arms)

    reward_history = np.zeros(num_dec, dtype=int)
    action_history = np.zeros(num_dec, dtype=int)

    for i in range(num_dec):

        action = policy(times, reward, esti_probs, rng, i, epsilon)

        current_reward = pull_arm(true_probs[action], reward_rng)

        reward[action] += current_reward
        times[action] += 1

        esti_probs[action] = reward[action] / times[action]

        reward_history[i] = current_reward
        action_history[i] = action

    return reward, times, reward_history, action_history


## method that runs multiple independent bandit simulations

def run_monte_carlo(M, num_dec, true_probs, policy, epsilon=0.1, seed=17):

    all_reward_history = np.zeros((M, num_dec))
    all_action_history = np.zeros((M, num_dec), dtype=int)

    master_seed = np.random.SeedSequence(seed)
    run_seeds = master_seed.spawn(M)

    for m in range(M):

        policy_seed, reward_seed = run_seeds[m].spawn(2)

        policy_rng = np.random.default_rng(policy_seed)
        reward_rng = np.random.default_rng(reward_seed)

        reward, times, reward_history, action_history = run_bandit(
            num_dec, true_probs, policy, policy_rng, epsilon, reward_rng=reward_rng
        )

        all_reward_history[m] = reward_history
        all_action_history[m] = action_history

    return all_reward_history, all_action_history


# setup

true_probs = np.array([0.3, 0.5, 0.7])
num_dec = 10000
epsilon = 0.05
M = 1000

dec_steps = np.arange(1, num_dec + 1)


# Monte Carlo simulations

## epsilon-greedy

all_reward_history, all_action_history = run_monte_carlo(M, num_dec, true_probs, policy_EG, epsilon=epsilon)

all_cumu_avg = np.cumsum(all_reward_history, axis=1) / dec_steps

mean_cumu_avg = np.mean(all_cumu_avg, axis=0)
std_cumu_avg = np.std(all_cumu_avg, axis=0, ddof=1)
se_cumu_avg = std_cumu_avg / np.sqrt(M)
ci95 = 1.96 * se_cumu_avg

print("EG final mean reward:", mean_cumu_avg[-1])


## random policy

all_reward_random, all_action_random = run_monte_carlo(M, num_dec, true_probs, policy_random)

all_cumu_avg_random = np.cumsum(all_reward_random, axis=1) / dec_steps

mean_cumu_avg_random = np.mean(all_cumu_avg_random, axis=0)
std_cumu_avg_random = np.std(all_cumu_avg_random, axis=0, ddof=1)
se_cumu_avg_random = std_cumu_avg_random / np.sqrt(M)
ci95_random = 1.96 * se_cumu_avg_random

print("Random final mean reward:", mean_cumu_avg_random[-1])


## UCB policy

all_reward_UCB, all_action_UCB = run_monte_carlo(M, num_dec, true_probs, policy_UCB)

all_cumu_avg_UCB = np.cumsum(all_reward_UCB, axis=1) / dec_steps

mean_cumu_avg_UCB = np.mean(all_cumu_avg_UCB, axis=0)
std_cumu_avg_UCB = np.std(all_cumu_avg_UCB, axis=0, ddof=1)
se_cumu_avg_UCB = std_cumu_avg_UCB / np.sqrt(M)
ci95_UCB = 1.96 * se_cumu_avg_UCB

print("UCB final mean reward:", mean_cumu_avg_UCB[-1])


## Thompson Sampling

all_reward_TS, all_action_TS = run_monte_carlo(M, num_dec, true_probs, policy_TS)

all_cumu_avg_TS = np.cumsum(all_reward_TS, axis=1) / dec_steps

mean_cumu_avg_TS = np.mean(all_cumu_avg_TS, axis=0)
std_cumu_avg_TS = np.std(all_cumu_avg_TS, axis=0, ddof=1)
se_cumu_avg_TS = std_cumu_avg_TS / np.sqrt(M)
ci95_TS = 1.96 * se_cumu_avg_TS

print("TS final mean reward:", mean_cumu_avg_TS[-1])


# calculate cumulative pseudo-regret

## random

all_regret_random = np.cumsum(np.max(true_probs) - true_probs[all_action_random], axis=1)
mean_regret_random = np.mean(all_regret_random, axis=0)


## epsilon-greedy

all_regret_EG = np.cumsum(np.max(true_probs) - true_probs[all_action_history], axis=1)
mean_regret_EG = np.mean(all_regret_EG, axis=0)


## UCB

all_regret_UCB = np.cumsum(np.max(true_probs) - true_probs[all_action_UCB], axis=1)
mean_regret_UCB = np.mean(all_regret_UCB, axis=0)


## Thompson Sampling

all_regret_TS = np.cumsum(np.max(true_probs) - true_probs[all_action_TS], axis=1)
mean_regret_TS = np.mean(all_regret_TS, axis=0)


print("Random final mean regret:", mean_regret_random[-1])
print("EG final mean regret:", mean_regret_EG[-1])
print("UCB final mean regret:", mean_regret_UCB[-1])
print("TS final mean regret:", mean_regret_TS[-1])


# compare average rewards

plt.figure(figsize=(10, 6))

plt.fill_between(dec_steps, mean_cumu_avg_random - ci95_random, mean_cumu_avg_random + ci95_random, alpha=0.12)
plt.fill_between(dec_steps, mean_cumu_avg - ci95, mean_cumu_avg + ci95, alpha=0.12)
plt.fill_between(dec_steps, mean_cumu_avg_UCB - ci95_UCB, mean_cumu_avg_UCB + ci95_UCB, alpha=0.12)
plt.fill_between(dec_steps, mean_cumu_avg_TS - ci95_TS, mean_cumu_avg_TS + ci95_TS, alpha=0.12)

plt.plot(dec_steps, mean_cumu_avg_random, label="Random")
plt.plot(dec_steps, mean_cumu_avg, label="Epsilon-Greedy")
plt.plot(dec_steps, mean_cumu_avg_UCB, label="UCB")
plt.plot(dec_steps, mean_cumu_avg_TS, label="Thompson Sampling")

plt.axhline(y=np.max(true_probs), color="black", linestyle="--", label="Optimal Expected Reward")

plt.xlabel("Decision")
plt.ylabel("Cumulative Average Reward")
plt.title("Bandit Policy Comparison: 1000 Monte Carlo Runs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("all_policies_comparison.png", dpi=300)
plt.show()


# compare cumulative regret

plt.figure(figsize=(10, 6))

plt.plot(dec_steps, mean_regret_random, label="Random")
plt.plot(dec_steps, mean_regret_EG, label="Epsilon-Greedy")
plt.plot(dec_steps, mean_regret_UCB, label="UCB")
plt.plot(dec_steps, mean_regret_TS, label="Thompson Sampling")

plt.xlabel("Decision")
plt.ylabel("Mean Cumulative Pseudo-Regret")
plt.title("Cumulative Regret Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("regret_comparison.png", dpi=300)
plt.show()


# compare the final average reward differences

## epsilon-greedy vs random

gap_history = all_cumu_avg - all_cumu_avg_random

mean_gap = np.mean(gap_history, axis=0)
std_gap = np.std(gap_history, axis=0, ddof=1)
se_gap = std_gap / np.sqrt(M)
ci95_gap = 1.96 * se_gap

print("Final EG vs Random gap:", mean_gap[-1])
print("95% CI:", mean_gap[-1] - ci95_gap[-1], mean_gap[-1] + ci95_gap[-1])


## UCB vs epsilon-greedy

gap_UCB_EG = all_cumu_avg_UCB - all_cumu_avg

mean_gap_UCB_EG = np.mean(gap_UCB_EG, axis=0)
std_gap_UCB_EG = np.std(gap_UCB_EG, axis=0, ddof=1)
se_gap_UCB_EG = std_gap_UCB_EG / np.sqrt(M)
ci95_gap_UCB_EG = 1.96 * se_gap_UCB_EG

print("Final UCB vs EG gap:", mean_gap_UCB_EG[-1])
print("95% CI:", mean_gap_UCB_EG[-1] - ci95_gap_UCB_EG[-1], mean_gap_UCB_EG[-1] + ci95_gap_UCB_EG[-1])


## TS vs UCB

gap_TS_UCB = all_cumu_avg_TS - all_cumu_avg_UCB

mean_gap_TS_UCB = np.mean(gap_TS_UCB, axis=0)
std_gap_TS_UCB = np.std(gap_TS_UCB, axis=0, ddof=1)
se_gap_TS_UCB = std_gap_TS_UCB / np.sqrt(M)
ci95_gap_TS_UCB = 1.96 * se_gap_TS_UCB

print("Final TS vs UCB gap:", mean_gap_TS_UCB[-1])
print("95% CI:", mean_gap_TS_UCB[-1] - ci95_gap_TS_UCB[-1], mean_gap_TS_UCB[-1] + ci95_gap_TS_UCB[-1])

# compare algorithms under different environments

scenarios = {
    "Easy": [0.3, 0.5, 0.7],
    "Medium": [0.4, 0.6, 0.7],
    "Hard": [0.60, 0.65, 0.70],
    "Very Hard": [0.68, 0.69, 0.70]
}

policies = {
    "Random": policy_random,
    "Epsilon-Greedy": policy_EG,
    "UCB": policy_UCB,
    "Thompson Sampling": policy_TS
}

M_test = 100
T_test = 3000

for scenario_name, probs in scenarios.items():

    probs = np.array(probs)

    print("\nScenario:", scenario_name)
    print("True probabilities:", probs)

    for policy_name, policy in policies.items():

        rewards, actions = run_monte_carlo(M_test, T_test, probs, policy, epsilon=epsilon)

        mean_reward = np.mean(rewards)

        final_regret = np.sum(np.max(probs) - probs[actions], axis=1)

        mean_regret = np.mean(final_regret)

        optimal_arm = np.argmax(probs)
        optimal_arm_rate = np.mean(actions == optimal_arm)

        print(policy_name, "Mean Reward:", round(mean_reward, 5), "Mean Regret:", round(mean_regret, 3), "Optimal Arm Rate:", round(optimal_arm_rate, 4))


