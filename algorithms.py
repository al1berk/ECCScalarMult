import time
import statistics
import matplotlib.pyplot as plt
from ecdsa import ellipticcurve, curves

# SMBR Function
def smbr_scalar_multiplication(k, point):
    result = ellipticcurve.INFINITY  # Initialize result to the point at infinity
    binary_k = bin(k)[2:]  # Convert scalar k to binary

    for bit in binary_k:
        result = result.double()  # Double the result
        if bit == '1':
            result += point  # Add the point if the bit is 1
    return result

# Montgomery Ladder Function
def montgomery_ladder(k, point):
    R0 = ellipticcurve.INFINITY  # Initialize R0 to the point at infinity
    R1 = point  # Initialize R1 to the given point
    binary_k = bin(k)[2:]  # Convert scalar k to binary

    for bit in binary_k:
        if bit == '1':
            R0, R1 = R1 + R0, R1.double()  # Perform specific operations if bit is 1
        else:
            R0, R1 = R0.double(), R0 + R1  # Perform different operations if bit is 0
    return R0

# Lookup Table Functions
def precompute_table(point, window_size):
    table = [point]  # Initialize the table with the given point
    for _ in range(1, 2 ** window_size):
        table.append(table[-1] + point)  # Precompute points for the table
    return table

def lookup_table_scalar_multiplication(k, point, window_size):
    table = precompute_table(point, window_size)  # Precompute the table
    result = ellipticcurve.INFINITY  # Initialize result to the point at infinity
    binary_k = bin(k)[2:]  # Convert scalar k to binary
    step = window_size  # Set the step size to the window size

    for i in range(0, len(binary_k), step):
        chunk = binary_k[i:i + step]  # Get the current chunk
        idx = int(chunk, 2)  # Convert the chunk to an integer

        for _ in range(step):
            result = result.double()  # Double the result for each bit in the chunk
        result += table[idx]  # Add the precomputed point to the result
    return result

def comb_method_low_latency(k, point, window_size=4):
    table = precompute_table(point, window_size)  # Precompute the table
    result = ellipticcurve.INFINITY  # Initialize result to the point at infinity
    binary_k = bin(k)[2:]  # Convert scalar k to binary
    padded_length = window_size * ((len(binary_k) + window_size - 1) // window_size)  # Calculate padded length
    padded_binary_k = binary_k.zfill(padded_length)  # Pad the binary representation
    windows = [int(''.join(x), 2) for x in zip(*[iter(padded_binary_k)] * window_size)]  # Split into windows

    for window in windows:
        for _ in range(window_size):
            result = result.double()  # Double the result for each bit in the window
        result += table[window]  # Add the precomputed point to the result
    return result

def measure_algorithms(scalar_values, window_size=4):
    curve = curves.SECP256k1.curve  # Get the SECP256k1 curve
    G = curves.SECP256k1.generator  # Get the generator point

    smbr_times = []
    montgomery_times = []
    lookup_times = []
    comb_times = []

    for scalar in scalar_values:
        start = time.time()
        _ = smbr_scalar_multiplication(scalar, G)
        smbr_times.append(time.time() - start)

        start = time.time()
        _ = montgomery_ladder(scalar, G)
        montgomery_times.append(time.time() - start)

        start = time.time()
        _ = lookup_table_scalar_multiplication(scalar, G, window_size)
        lookup_times.append(time.time() - start)

        start = time.time()
        _ = comb_method_low_latency(scalar, G, window_size)
        comb_times.append(time.time() - start)

    results = {
        'SMBR': statistics.mean(smbr_times),
        'Montgomery': statistics.mean(montgomery_times),
        'LookupTable': statistics.mean(lookup_times),
        'CombMethod': statistics.mean(comb_times)
    }
    return results

def visualize_results(scalar_sets, window_size=4):
    result_list = []
    labels = []

    for idx, scalars in enumerate(scalar_sets, start=1):
        results = measure_algorithms(scalars, window_size)
        result_list.append(results)
        labels.append(f"Set {idx}")
        print(
            f"Scalar Set {idx} --> SMBR: {results['SMBR']:.6f} s, "
            f"Montgomery: {results['Montgomery']:.6f} s, "
            f"LookupTable: {results['LookupTable']:.6f} s, "
            f"CombMethod: {results['CombMethod']:.6f} s"
        )

    x_positions = range(len(scalar_sets))
    width = 0.2

    smbr_means = [res['SMBR'] for res in result_list]
    monty_means = [res['Montgomery'] for res in result_list]
    lookup_means = [res['LookupTable'] for res in result_list]
    comb_means = [res['CombMethod'] for res in result_list]

    # Bar Chart
    plt.figure(figsize=(14, 10))  # Increased figure size for better readability
    plt.bar([x - 1.5 * width for x in x_positions], smbr_means, width=width, color='tab:blue', label='SMBR')
    plt.bar([x - 0.5 * width for x in x_positions], monty_means, width=width, color='tab:green', label='Montgomery')
    plt.bar([x + 0.5 * width for x in x_positions], lookup_means, width=width, color='tab:orange', label='Lookup Table')
    plt.bar([x + 1.5 * width for x in x_positions], comb_means, width=width, color='tab:red', label='Comb Method')

    plt.xticks(x_positions, labels, fontsize=20)  # Increased tick label fontsize
    plt.yticks(fontsize=18)  # Increased y-tick label fontsize
    plt.ylabel('Average Execution Time (seconds)', fontsize=22)  # Increased y-label fontsize
    plt.title('Comparison of ECC Scalar Multiplication Methods - Bar Chart', fontsize=26)  # Increased title fontsize
    plt.legend(fontsize=20)  # Increased legend fontsize
    plt.tight_layout()
    plt.show()

    # Line Chart
    plt.figure(figsize=(14, 10))  # Increased figure size for better readability
    plt.plot(x_positions, smbr_means, marker='o', color='tab:blue', linestyle='--', label='SMBR')
    plt.plot(x_positions, monty_means, marker='o', color='tab:green', linestyle='--', label='Montgomery')
    plt.plot(x_positions, lookup_means, marker='o', color='tab:orange', linestyle='--', label='Lookup Table')
    plt.plot(x_positions, comb_means, marker='o', color='tab:red', linestyle='--', label='Comb Method')

    plt.xticks(x_positions, labels, fontsize=20)  # Increased tick label fontsize
    plt.yticks(fontsize=18)  # Increased y-tick label fontsize
    plt.ylabel('Average Execution Time (seconds)', fontsize=22)  # Increased y-label fontsize
    plt.title('Comparison of ECC Scalar Multiplication Methods - Line Chart', fontsize=26)  # Increased title fontsize
    plt.legend(fontsize=20)  # Increased legend fontsize
    plt.tight_layout()
    plt.show()

def measure_algorithms_with_variance(scalar_values, window_size=4):
    curve = curves.SECP256k1.curve  # Get the SECP256k1 curve
    G = curves.SECP256k1.generator  # Get the generator point

    smbr_times = []
    montgomery_times = []
    lookup_times = []
    comb_times = []

    for scalar in scalar_values:
        start = time.time()
        _ = smbr_scalar_multiplication(scalar, G)
        smbr_times.append(time.time() - start)

        start = time.time()
        _ = montgomery_ladder(scalar, G)
        montgomery_times.append(time.time() - start)

        start = time.time()
        _ = lookup_table_scalar_multiplication(scalar, G, window_size)
        lookup_times.append(time.time() - start)

        start = time.time()
        _ = comb_method_low_latency(scalar, G, window_size)
        comb_times.append(time.time() - start)

    results = {
        'SMBR': {'mean': statistics.mean(smbr_times), 'std': statistics.stdev(smbr_times) if len(smbr_times) > 1 else 0},
        'Montgomery': {'mean': statistics.mean(montgomery_times), 'std': statistics.stdev(montgomery_times) if len(montgomery_times) > 1 else 0},
        'LookupTable': {'mean': statistics.mean(lookup_times), 'std': statistics.stdev(lookup_times) if len(lookup_times) > 1 else 0},
        'CombMethod': {'mean': statistics.mean(comb_times), 'std': statistics.stdev(comb_times) if len(comb_times) > 1 else 0}
    }
    return results

def visualize_histograms(scalar_sets, window_size=4):
    for idx, scalars in enumerate(scalar_sets, start=1):
        curve = curves.SECP256k1.curve
        G = curves.SECP256k1.generator

        smbr_times = []
        montgomery_times = []
        lookup_times = []
        comb_times = []

        for scalar in scalars:
            start = time.time()
            _ = smbr_scalar_multiplication(scalar, G)
            smbr_times.append(time.time() - start)

            start = time.time()
            _ = montgomery_ladder(scalar, G)
            montgomery_times.append(time.time() - start)

            start = time.time()
            _ = lookup_table_scalar_multiplication(scalar, G, window_size)
            lookup_times.append(time.time() - start)

            start = time.time()
            _ = comb_method_low_latency(scalar, G, window_size)
            comb_times.append(time.time() - start)

        plt.figure(figsize=(14, 10))  # Increased figure size for better readability
        plt.hist(smbr_times, bins=15, alpha=0.5, label='SMBR', color='tab:blue')
        plt.hist(montgomery_times, bins=15, alpha=0.5, label='Montgomery', color='tab:green')
        plt.hist(lookup_times, bins=15, alpha=0.5, label='Lookup Table', color='tab:orange')
        plt.hist(comb_times, bins=15, alpha=0.5, label='Comb Method', color='tab:red')

        plt.title(f'Distribution of ECC Scalar Multiplication Methods - Set {idx}', fontsize=26)  # Increased title fontsize
        plt.xlabel('Execution Time (seconds)', fontsize=22)  # Increased x-label fontsize
        plt.ylabel('Frequency', fontsize=22)  # Increased y-label fontsize
        plt.xticks(fontsize=20)  # Increased x-tick label fontsize
        plt.yticks(fontsize=20)  # Increased y-tick label fontsize
        plt.legend(fontsize=20)  # Increased legend fontsize
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    scalar_set_1 = [10_000, 20_000, 30_000]
    scalar_set_2 = [100_000, 200_000, 300_000]
    scalar_set_3 = [999_999, 1_234_567, 2_000_000]

    all_scalar_sets = [scalar_set_1, scalar_set_2, scalar_set_3]

    visualize_results(all_scalar_sets, window_size=4)
    visualize_histograms(all_scalar_sets, window_size=4)
