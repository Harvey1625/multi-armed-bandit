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

def run_bandit(num_dec, true_probs, policy, rng, epsilon=0.1,
               reward_rng=None, changing_probs=None, window=None):

    num_arms = len(true_probs)

    if reward_rng is None:
        reward_rng = rng

    reward = np.zeros(num_arms, dtype=int)
    times = np.zeros(num_arms, dtype=int)
    esti_probs = np.zeros(num_arms)

    reward_history = np.zeros(num_dec, dtype=int)
    action_history = np.zeros(num_dec, dtype=int)

    for i in range(num_dec):


        if window is not None and i > window:
            old_i = i - window - 1
            old_action = action_history[old_i]

            reward[old_action] -= reward_history[old_i]
            times[old_action] -= 1

            if times[old_action] > 0:
                esti_probs[old_action] = (
                    reward[old_action] / times[old_action]
                )
            else:
                esti_probs[old_action] = 0


        if window is None:
            action = policy(times, reward, esti_probs, rng, i, epsilon)

        else:
            policy_i = min(i, window)

            if policy is policy_UCB and np.any(times == 0):
                action = np.flatnonzero(times == 0)[0]
            else:
                action = policy(
                    times, reward, esti_probs, rng, policy_i, epsilon
                )


        if changing_probs is None:
            current_prob = true_probs[action]
        else:
            current_prob = changing_probs[i, action]

        current_reward = pull_arm(current_prob, reward_rng)

        reward[action] += current_reward
        times[action] += 1

        esti_probs[action] = reward[action] / times[action]

        reward_history[i] = current_reward
        action_history[i] = action


    total_times = np.bincount(action_history, minlength=num_arms)
    total_reward = np.bincount(
        action_history,
        weights=reward_history,
        minlength=num_arms
    ).astype(int)

    return total_reward, total_times, reward_history, action_history


## method that runs multiple independent bandit simulations

def run_monte_carlo(M, num_dec, true_probs, policy, epsilon=0.1,
                    seed=17, changing_probs=None, window=None):

    all_reward_history = np.zeros((M, num_dec))
    all_action_history = np.zeros((M, num_dec), dtype=int)

    master_seed = np.random.SeedSequence(seed)
    run_seeds = master_seed.spawn(M)

    for m in range(M):

        policy_seed, reward_seed = run_seeds[m].spawn(2)

        policy_rng = np.random.default_rng(policy_seed)
        reward_rng = np.random.default_rng(reward_seed)

        reward, times, reward_history, action_history = run_bandit(
            num_dec,
            true_probs,
            policy,
            policy_rng,
            epsilon,
            reward_rng=reward_rng,
            changing_probs=changing_probs,
            window=window
        )

        all_reward_history[m] = reward_history
        all_action_history[m] = action_history

        if (m + 1) % 100 == 0:
            print(f"Completed {m + 1}/{M} runs", flush=True)

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


# non-stationary environment

change_point = num_dec // 2

changing_probs = np.zeros((num_dec, len(true_probs)))

changing_probs[:change_point] = true_probs
changing_probs[change_point:] = [0.7, 0.5, 0.3]

M_change = 1000


# UCB

rewards_change_UCB, actions_change_UCB = run_monte_carlo(
    M_change,
    num_dec,
    true_probs,
    policy_UCB,
    changing_probs=changing_probs
)


# Thompson Sampling

rewards_change_TS, actions_change_TS = run_monte_carlo(
    M_change,
    num_dec,
    true_probs,
    policy_TS,
    changing_probs=changing_probs
)


# dynamic pseudo-regret

step_index = np.arange(num_dec)

best_probs = np.max(changing_probs, axis=1)

selected_probs_UCB = changing_probs[step_index, actions_change_UCB]
selected_probs_TS = changing_probs[step_index, actions_change_TS]

regret_change_UCB = np.cumsum(
    best_probs - selected_probs_UCB,
    axis=1
)

regret_change_TS = np.cumsum(
    best_probs - selected_probs_TS,
    axis=1
)

mean_regret_change_UCB = np.mean(regret_change_UCB, axis=0)
mean_regret_change_TS = np.mean(regret_change_TS, axis=0)

print("Non-stationary UCB regret:", mean_regret_change_UCB[-1])
print("Non-stationary TS regret:", mean_regret_change_TS[-1])

# sliding-window experiments and saved results

from pathlib import Path
import csv

results_dir = Path(__file__).resolve().parent / "bandit_results"
results_dir.mkdir(exist_ok=True)


# Save the full-history baseline results

np.savez_compressed(
    results_dir / "baseline_results.npz",
    changing_probs=changing_probs,
    rewards_UCB=rewards_change_UCB,
    actions_UCB=actions_change_UCB,
    rewards_TS=rewards_change_TS,
    actions_TS=actions_change_TS
)


# Summarize one method

def summarize_method(actions):
    selected_probs = changing_probs[step_index, actions]

    cumulative_regret = np.cumsum(
        best_probs - selected_probs,
        axis=1
    )

    mean_regret = np.mean(cumulative_regret, axis=0)

    ci_regret = (
        1.96
        * np.std(cumulative_regret, axis=0, ddof=1)
        / np.sqrt(actions.shape[0])
    )

    post_regret = (
        cumulative_regret[:, -1]
        - cumulative_regret[:, change_point - 1]
    )

    # After the switch, arm 0 is the best arm
    post_actions = actions[:, change_point:]

    # Average within each 100-step block, then across runs
    bin_size = 100
    bin_starts = np.arange(0, post_actions.shape[1], bin_size)

    rates_by_run = np.column_stack([
        np.mean(post_actions[:, start:start + bin_size] == 0, axis=1)
        for start in bin_starts
    ])

    mean_rate = np.mean(rates_by_run, axis=0)

    ci_rate = (
        1.96
        * np.std(rates_by_run, axis=0, ddof=1)
        / np.sqrt(actions.shape[0])
    )

    return {
        "mean_regret": mean_regret,
        "ci_regret": ci_regret,
        "post_regret": np.mean(post_regret),
        "post_optimal_rate": np.mean(post_actions == 0),
        "bin_steps": bin_starts + 1,
        "mean_rate": mean_rate,
        "ci_rate": ci_rate
    }


comparison = {
    "UCB": summarize_method(actions_change_UCB),
    "TS": summarize_method(actions_change_TS)
}


# Run the sliding-window versions

window_lengths = [100, 500, 1000]

window_policies = {
    "UCB": policy_UCB,
    "TS": policy_TS
}

for window in window_lengths:

    for policy_name, policy in window_policies.items():

        label = f"{policy_name} W={window}"

        cache_file = results_dir / (
            f"{policy_name}_W{window}_M{M_change}_T{num_dec}_seed17.npz"
        )

        if cache_file.exists():

            print("Loading:", label, flush=True)

            with np.load(cache_file) as saved:
                if not np.array_equal(
                    saved["changing_probs"], changing_probs
                ):
                    raise ValueError(
                        "Environment changed: remove the old cache file."
                    )

                actions_window = saved["actions"]

        else:

            print("Running:", label, flush=True)

            rewards_window, actions_window = run_monte_carlo(
                M_change,
                num_dec,
                true_probs,
                policy,
                seed=17,
                changing_probs=changing_probs,
                window=window
            )

            np.savez_compressed(
                cache_file,
                changing_probs=changing_probs,
                rewards=rewards_window,
                actions=actions_window
            )

            del rewards_window

        comparison[label] = summarize_method(actions_window)

        result = comparison[label]

        print(
            label,
            "Mean regret:",
            round(result["mean_regret"][-1], 4),
            "95% CI half-width:",
            round(result["ci_regret"][-1], 4),
            flush=True
        )

        del actions_window


# Save the summary table

with open(
    results_dir / "summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "method",
        "mean_total_regret",
        "ci95_low",
        "ci95_high",
        "mean_post_switch_regret",
        "post_switch_optimal_arm_rate"
    ])

    for label, result in comparison.items():

        mean = result["mean_regret"][-1]
        half_width = result["ci_regret"][-1]

        writer.writerow([
            label,
            mean,
            mean - half_width,
            mean + half_width,
            result["post_regret"],
            result["post_optimal_rate"]
        ])


# Plot cumulative dynamic pseudo-regret

plt.figure(figsize=(10, 6))

for label, result in comparison.items():

    line, = plt.plot(
        dec_steps,
        result["mean_regret"],
        label=label
    )

    plt.fill_between(
        dec_steps,
        result["mean_regret"] - result["ci_regret"],
        result["mean_regret"] + result["ci_regret"],
        color=line.get_color(),
        alpha=0.1
    )

plt.axvline(
    change_point + 0.5,
    color="black",
    linestyle="--",
    label="Environment switch"
)

plt.xlabel("Decision")
plt.ylabel("Mean Cumulative Dynamic Pseudo-Regret")
plt.title("Full History vs Sliding Windows")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "nonstationary_regret.png",
    dpi=300
)
plt.close()


# Plot adaptation after the switch

plt.figure(figsize=(10, 6))

for label, result in comparison.items():

    line, = plt.plot(
        result["bin_steps"],
        result["mean_rate"],
        label=label
    )

    plt.fill_between(
        result["bin_steps"],
        np.maximum(0, result["mean_rate"] - result["ci_rate"]),
        np.minimum(1, result["mean_rate"] + result["ci_rate"]),
        color=line.get_color(),
        alpha=0.1
    )

plt.xlabel("Decision After Switch (100-Step Bin Start)")
plt.ylabel("New Best Arm Selection Rate")
plt.title("Adaptation After the Environment Switch")
plt.ylim(0, 1.02)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "adaptation_after_switch.png",
    dpi=300
)
plt.close()

print("All experiments finished. Results saved to:", results_dir)