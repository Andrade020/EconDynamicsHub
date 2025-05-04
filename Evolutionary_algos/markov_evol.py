import numpy as np
import matplotlib.pyplot as plt

# =======================
# Configuration Parameters
# =======================
num_steps               = 1000    # total simulation steps
initial_population      = 1000    # starting total number of organisms
initial_diversity       = 10     # number of distinct genotypes at start
state_count             = 3      # number of states in the Markov chain

# Global Markov chain transition matrix (rows sum to 1)
rng = np.random.default_rng(42)
raw = rng.random((state_count, state_count))
global_transition = raw / raw.sum(axis=1, keepdims=True)

death_threshold         = 3      # incorrect in a row before death
mutation_threshold      = 3      # correct in a row before mutation
mutation_scale          = 0.2    # perturbation scale for mutation
duplication_probability = 0.410   # chance for each correct to duplicate

global_states = list(range(state_count))


# =======================
# Helper Functions
# =======================
def normalize_rows(M: np.ndarray) -> np.ndarray:
    M = np.maximum(M, 0)
    sums = M.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return M / sums

def mutate_matrix(P: np.ndarray) -> np.ndarray:
    P_new = P.copy()
    row = rng.integers(0, P.shape[0])
    noise = rng.standard_normal(P.shape[1]) * mutation_scale
    P_new[row] += noise
    return normalize_rows(P_new)

def make_genotype_key(P: np.ndarray, cc: int, ci: int) -> tuple:
    flat = tuple(np.round(P.flatten(), 4))
    return (flat, cc, ci)


# =======================
# Initialize Aggregated Population with Multiple Starting Genotypes
# =======================
population = {}
per_genotype = initial_population // initial_diversity

for i in range(initial_diversity):
    # start each genotype as a noisy variant of the global_transition
    Pi = normalize_rows(global_transition + rng.standard_normal(global_transition.shape)*0.1)
    key = make_genotype_key(Pi, cc=0, ci=0)
    population[key] = {
        'P': Pi,
        'cc': 0,
        'ci': 0,
        'count': per_genotype
    }

# if there's remainder, add to the first genotype
remainder = initial_population - per_genotype * initial_diversity
if remainder > 0:
    first_key = next(iter(population))
    population[first_key]['count'] += remainder

# Time series storage
time_series   = {}   # key -> list of counts over time
history_total = []   # total population over time


# =======================
# Setup Real-Time Plot with 3 Panels
# =======================
plt.ion()
fig, (ax_ts, ax_glob, ax_top1) = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: time series
ax_ts.set_xlabel('Step')
ax_ts.set_ylabel('Count')
ax_ts.set_title('Top 4 Genotypes & Others')
ax_ts.set_xlim(0, num_steps)
ax_ts.set_ylim(0, initial_population * 2)

# Panel 2: global matrix
im_glob = ax_glob.imshow(global_transition, cmap = 'viridis', vmin=0, vmax=1)
ax_glob.set_title('Global Transition Matrix')
ax_glob.set_xticks(range(state_count))
ax_glob.set_yticks(range(state_count))

# Panel 3: top-1 matrix (placeholder)
P_placeholder = next(iter(population.values()))['P']
im_top1 = ax_top1.imshow(P_placeholder,cmap = 'viridis', vmin=0, vmax=1)
ax_top1.set_title('Top-1 Genotype Matrix\nCount=0')
ax_top1.set_xticks(range(state_count))
ax_top1.set_yticks(range(state_count))

plt.tight_layout()


# =======================
# Simulation Loop with Real-Time Updates
# =======================
current_state = rng.choice(global_states)

for step in range(num_steps):
    next_state = rng.choice(global_states, p=global_transition[current_state])
    new_pop = {}

    # — Process each genotype group —
    for key, info in population.items():
        P, cc, ci, n = info['P'], info['cc'], info['ci'], info['count']
        p_corr = P[current_state, next_state]
        k = rng.binomial(n, p_corr)  # correct
        m = n - k                    # incorrect

        # correct: survivors + stochastic duplicates + possible mutants
        if k > 0:
            cc_new, ci_new = cc + 1, 0
            survivors = k
            offspring = rng.binomial(k, duplication_probability)
            mutants = rng.binomial(k, duplication_probability) if cc_new >= mutation_threshold else 0

            # add survivors
            if survivors:
                kk = make_genotype_key(P, cc_new, ci_new)
                new_pop.setdefault(kk, {'P': P, 'cc': cc_new, 'ci': ci_new, 'count': 0})
                new_pop[kk]['count'] += survivors

            # add duplicates
            if offspring:
                new_pop[kk]['count'] += offspring

            # add mutants
            if mutants:
                Pm = mutate_matrix(P)
                mk = make_genotype_key(Pm, 0, 0)
                new_pop.setdefault(mk, {'P': Pm, 'cc': 0, 'ci': 0, 'count': 0})
                new_pop[mk]['count'] += mutants

        # incorrect: survive until threshold
        if m > 0:
            cc_bad, ci_bad = 0, ci + 1
            if ci_bad < death_threshold:
                kk = make_genotype_key(P, cc_bad, ci_bad)
                new_pop.setdefault(kk, {'P': P, 'cc': cc_bad, 'ci': ci_bad, 'count': 0})
                new_pop[kk]['count'] += m

    population    = new_pop
    current_state = next_state

    # — Record counts —
    counts = { key: info['count'] for key, info in population.items() }
    total  = sum(counts.values())
    history_total.append(total)

    # — Synchronize time_series lengths —
    for key in list(time_series.keys()):
        while len(time_series[key]) < step:
            time_series[key].append(0)
    for key in counts:
        if key not in time_series:
            time_series[key] = [0] * step
    for key in time_series:
        time_series[key].append(counts.get(key, 0))

    # — Identify top4 & compute 'others' —
    top4     = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:4]
    top_keys = [k for k, _ in top4]
    others_ts = [
        history_total[i] - sum(time_series[k][i] for k in top_keys)
        for i in range(step + 1)
    ]

    # — Update time-series panel —
    ax_ts.clear()
    for idx, k in enumerate(top_keys, 1):
        ax_ts.plot(range(step+1), time_series[k], label=f'Gen {idx}')
    ax_ts.plot(range(step+1), others_ts, '--', label='Others')
    ax_ts.set_xlim(0, step+1)
    ax_ts.set_ylim(0, max(history_total)*1.1)
    ax_ts.set_title('Top 4 Genotypes & Others')
    ax_ts.legend(loc='upper left')

    # — Update Top-1 matrix panel —
    if top_keys:
        top1_key = top_keys[0]
        flat, _, _ = top1_key
        P_top1 = np.array(flat).reshape((state_count, state_count))
        im_top1.set_data(P_top1)
        im_top1.set_clim(0.0, 1.0)
        ax_top1.set_title(f'Top-1 Genotype Matrix\nCount={counts[top1_key]}')

    # — Redraw all panels —
    plt.draw()
    plt.pause(0.01)

    print(f"Step {step:3d}: total={total}, top1_count={counts.get(top1_key, 0)}")
    if total == 0:
        print("All organisms have died.")
        break

plt.ioff()
plt.show()
