
import math
import statistics


def _permutation_friedman(matrix, *, permutations, rng):
    observed = _friedman_statistic(matrix)
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if not matrix or observed == 0.0:
        return observed, 1.0, 0.0

    extreme = 0
    for _ in range(permutations):
        permuted = []
        for row in matrix:
            shuffled = list(row)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        if _friedman_statistic(permuted) >= observed - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)
    kendalls_w = observed / (n * (k - 1))
    return observed, p_value, kendalls_w

def _friedman_statistic(matrix):
    if not matrix:
        return 0.0
    n = len(matrix)
    k = len(matrix[0])
    ranked = [_average_ranks(row) for row in matrix]
    rank_sums = [sum(row[column] for row in ranked) for column in range(k)]
    statistic = (
        12.0 * sum(value * value for value in rank_sums) / (n * k * (k + 1))
        - 3.0 * n * (k + 1)
    )
    tie_sum = 0
    for row in matrix:
        counts = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_sum += sum(count ** 3 - count for count in counts.values())
    correction = 1.0 - tie_sum / (n * (k ** 3 - k))
    return 0.0 if correction == 0 else statistic / correction

def _average_ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return ranks

def _wilcoxon_signed_rank(differences, *, permutations, rng):
    rounded = [round(float(value), 12) for value in differences]
    nonzero = [value for value in rounded if value != 0.0]
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial_correlation": 0.0,
            "n_nonzero_pairs": 0,
            "method": "all_zero",
        }
    ranks = _average_ranks([abs(value) for value in nonzero])
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    observed_distance = abs(positive - (positive + negative) / 2.0)
    if len(ranks) <= 20:
        extreme = 0
        total = 1 << len(ranks)
        for mask in range(total):
            permuted_positive = sum(
                rank for index, rank in enumerate(ranks) if mask & (1 << index)
            )
            distance = abs(permuted_positive - (positive + negative) / 2.0)
            if distance >= observed_distance - 1e-12:
                extreme += 1
        p_value = extreme / total
        method = "exact_sign_flip"
    else:
        extreme = 0
        rank_total = positive + negative
        for _ in range(permutations):
            permuted_positive = sum(rank for rank in ranks if rng.random() < 0.5)
            distance = abs(permuted_positive - rank_total / 2.0)
            if distance >= observed_distance - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        method = "monte_carlo_sign_flip"
    return {
        "statistic": min(positive, negative),
        "p_value": p_value,
        "rank_biserial_correlation": (positive - negative) / (positive + negative),
        "n_nonzero_pairs": len(nonzero),
        "method": method,
    }

def _holm_adjust(p_values):
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted

def _bootstrap_median_ci(differences, *, resamples, rng):
    if not differences:
        return math.nan, math.nan
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(differences) for _ in differences]
        estimates.append(statistics.median(sample))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)

def _bootstrap_statistic_ci(values, *, statistic, resamples, rng):
    if not values:
        return "", ""
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(values) for _ in values]
        estimates.append(statistic(sample))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)

def _pearson_asymmetry(values):
    if len(values) < 3:
        return 0.0
    spread = statistics.pstdev(values)
    if spread == 0.0:
        return 0.0
    return 3.0 * (statistics.mean(values) - statistics.median(values)) / spread

def _exact_sign_test(differences):
    nonzero = [value for value in differences if value != 0]
    count = len(nonzero)
    if count == 0:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    tail = min(positive, count - positive)
    probability = sum(math.comb(count, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** count))

def _quantile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
