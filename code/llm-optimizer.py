#!/usr/bin/env python3
"""
Simplified DoH Trace Synthesis and Optimization Script
"""

import time
import scripts.init_dataset as init_dataset
from API_KEY import GEMINI_API_KEYS
import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from google import genai
import warnings
warnings.filterwarnings('ignore')


# Configuration
CONFIG = {
    'source_location': 'LOC2',
    'target_location': 'LOC3',
    'num_sample_pairs': 20,
    'num_test_traces': 5,
    'max_iterations': 100,
    'model_name': "gemma-3-27b-it",
    'data_path': "../dataset/processed/LOC2-LOC3-Date-ID.csv",
    'data_points': 64,  # Number of data points in each trace
    'max_retries': 3    # Maximum retries for parsing/validation
}

# Global variables for API key management
current_key_index = 0
clients = {}


def initialize_clients():
    """Initialize Gemini clients for all API keys"""
    global clients
    if not isinstance(GEMINI_API_KEYS, list):
        raise ValueError("GEMINI_API_KEYS should be a list of API keys")

    print(f"Initializing {len(GEMINI_API_KEYS)} Gemini clients...")
    for i, api_key in enumerate(GEMINI_API_KEYS):
        clients[i] = genai.Client(api_key=api_key)
    print(f"Successfully initialized {len(clients)} clients")


def get_next_client():
    """Get the next client using round-robin selection"""
    global current_key_index
    client = clients[current_key_index]
    current_key_index = (current_key_index + 1) % len(clients)
    return client, current_key_index - 1 if current_key_index > 0 else len(clients) - 1


def load_data():
    """Load and preprocess the dataset"""
    print("Loading Dataset...")
    locations = [CONFIG['source_location'], CONFIG['target_location']]

    # Load the dataset
    df = pd.read_csv(CONFIG['data_path'])
    df.drop(['Date', 'ID'], inplace=True, axis=1)
    # Keep only relevant columns (2 for Website/Location + data_points)
    df = df.iloc[:, :CONFIG['data_points'] + 2]

    # Get train-test split
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    print(f"Dataset loaded successfully. Train shape: {train_df.shape}")
    return train_df, test_df


def get_random_pairs(df, iteration):
    """Get random pairs of traces from source and target locations"""
    np.random.seed(iteration)  # For reproducibility

    # Find common websites
    source_websites = set(
        df[df['Location'] == CONFIG['source_location']]['Website'])
    target_websites = set(
        df[df['Location'] == CONFIG['target_location']]['Website'])
    common_websites = list(source_websites & target_websites)

    if len(common_websites) < CONFIG['num_sample_pairs']:
        raise ValueError(
            f"Not enough common websites to sample {CONFIG['num_sample_pairs']} pairs.")

    # Select random websites
    selected_websites = np.random.choice(
        common_websites, size=CONFIG['num_sample_pairs'], replace=False)

    pairs = []
    for web_id in selected_websites:
        source_samples = df[(df['Website'] == web_id) & (
            df['Location'] == CONFIG['source_location'])]
        target_samples = df[(df['Website'] == web_id) & (
            df['Location'] == CONFIG['target_location'])]

        sample1 = source_samples.sample(n=1).iloc[0, 2:].values
        sample2 = target_samples.sample(n=1).iloc[0, 2:].values

        pairs.append((sample1.tolist(), sample2.tolist()))

    return pairs


def compute_mse(df, trace, website_id):
    """Compute MSE between trace and target location median"""
    candidates = df[(df['Website'] == website_id) & (
        df['Location'] == CONFIG['target_location'])]

    if len(candidates) == 0:
        return float('inf')

    target_values = candidates.iloc[:, 2:].median(axis=0).values
    mse = np.mean((np.array(trace) - target_values) ** 2)
    return mse


def create_initial_prompt(sample_pairs, source_traces):
    """Create initial synthesis prompt"""
    pairs_str = "\n".join([f"({pair[0]}, {pair[1]})" for pair in sample_pairs])
    traces_str = "\n".join([f"{trace}" for trace in source_traces])

    return f"""You are given packet counts of a DoH trace from a source location. Your task is to generate a DoH trace for the target location based on the source location's trace.
Positive values indicate uploads, while negative values indicate downloads. Each trace contains {CONFIG['data_points']} values.

Here are some actual pairs of traces from the source and target locations:
{pairs_str}

Let's slowly change the following traces to the target location, by modifying the source trace slightly:
{traces_str}

You should give the new traces in python in the format synthesized: List[List[int]]. Do not write or execute any code.
synthesized = [<trace1>, <trace2>, ...]. Think step-by-step, """


def create_optimizer_prompt(sample_pairs, current_synthesis_with_errors):
    """Create optimization prompt"""
    pairs_str = "\n".join([f"({pair[0]}, {pair[1]})" for pair in sample_pairs])

    return f"""You are given packet counts of a DoH trace from a source location. Your task is to generate a DoH trace for the target location based on the source location's trace.
Positive values indicate uploads, while negative values indicate downloads. Each trace contains {CONFIG['data_points']} values.

Here are some actual pairs of traces from the source and target locations:
{pairs_str}

Here are the traces you generated, along with their errors to the target location, lower errors indicate better traces:
(Original Source, Best Synthesis, Current Synthesis), Error
{current_synthesis_with_errors}

Change the current synthesis traces slightly to reduce the errors, new traces should be different to the best synthesis.
Output in python block in the format synthesized: List[List[int]]. Do not write or execute any code.
synthesized = [<trace1>, <trace2>, ...]. Think step-by-step, """


def parse_response(response_text):
    """Parse synthesized traces from AI response"""
    # Try to extract python code block
    code_block_match = re.search(
        r"```python(.*?)```", response_text, re.DOTALL)
    if code_block_match:
        code_block = code_block_match.group(1)
    else:
        code_block = response_text

    # Extract synthesized assignment
    synth_match = re.search(r"synthesized\s*=\s*(\[[\s\S]*\])", code_block)
    if not synth_match:
        raise ValueError("No synthesized list found in response.")

    synth_str = synth_match.group(1)

    # Clean up the string
    synth_str = re.sub(r',\s*\]', ']', synth_str)
    synth_str = re.sub(r',\s*\n\s*\]', ']', synth_str)
    synth_str = synth_str.strip()

    # Balance brackets
    open_brackets = synth_str.count('[')
    close_brackets = synth_str.count(']')
    if open_brackets > close_brackets:
        synth_str += ']' * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        synth_str = synth_str.rstrip(']') + ']' * open_brackets

    # Parse the list
    synthesized = ast.literal_eval(synth_str)
    return synthesized


def validate_traces(traces):
    """Validate that traces have correct number of data points"""
    if not isinstance(traces, list):
        raise ValueError(f"Expected list, got {type(traces)}")

    for i, trace in enumerate(traces):
        if not isinstance(trace, list):
            raise ValueError(f"Trace {i} is not a list: {type(trace)}")

        if len(trace) != CONFIG['data_points']:
            raise ValueError(
                f"Trace {i} has {len(trace)} points, expected {CONFIG['data_points']}")

        # Check if all elements are numbers
        for j, val in enumerate(trace):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Trace {i}, element {j} is not a number: {val}")

    return True


def get_ai_response_with_retry(prompt, retry_context=""):
    """Get AI response with retry logic for parsing/validation failures"""
    for attempt in range(CONFIG['max_retries']):
        try:
            # Get next client using round-robin
            client, key_index = get_next_client()

            print(
                f"AI request attempt {attempt + 1}/{CONFIG['max_retries']} {retry_context} (using API key {key_index + 1}/{len(clients)})")

            response = client.models.generate_content(
                model=CONFIG['model_name'],
                contents=prompt
            )

            if attempt == 0:  # Print response only on first attempt
                print("AI Response:")
                print(
                    response.text[:500] + "..." if len(response.text) > 500 else response.text)

            # Parse response
            print(response.text)
            synthesized = parse_response(response.text)

            # Validate traces
            validate_traces(synthesized)

            print(
                f"✓ Successfully parsed and validated {len(synthesized)} traces (API key {key_index + 1})")
            return synthesized

        except Exception as e:
            print(
                f"✗ Attempt {attempt + 1} failed with API key {key_index + 1}: {e}")
            if attempt == CONFIG['max_retries'] - 1:
                print("All retry attempts failed!")
                raise e

            # Modify prompt slightly for retry
            prompt += f"\n\nIMPORTANT: Please ensure each trace has exactly {CONFIG['data_points']} integer values."

        time.sleep(10)  # Wait before retrying

    return None


def initial_synthesis(train_df):
    """Perform initial trace synthesis"""
    print("\n=== INITIAL SYNTHESIS ===")

    # Get source test data
    source_test_df = train_df[train_df['Location'] == CONFIG['source_location']].sample(
        n=CONFIG['num_test_traces'], random_state=42
    )

    source_websites = source_test_df['Website'].values
    source_traces = source_test_df.iloc[:, 2:].values.tolist()

    # Get sample pairs
    sample_pairs = get_random_pairs(train_df, 0)

    # Create prompt and get response
    prompt = create_initial_prompt(sample_pairs, source_traces)
    synthesized = get_ai_response_with_retry(prompt, "(initial synthesis)")

    # Calculate errors
    errors = []
    for trace, website_id in zip(synthesized, source_websites):
        error = compute_mse(train_df, trace, website_id)
        errors.append(error)

    print(f"Initial synthesis complete. Mean error: {np.mean(errors):.4f}")
    return synthesized, errors, source_test_df


def optimize_iteration(train_df, source_test_df, current_synthesized, current_errors, iteration, best_synthesized, best_errors):
    """Run one optimization iteration"""
    print(f"\n=== ITERATION {iteration + 1}/{CONFIG['max_iterations']} ===")

    # Get fresh sample pairs
    sample_pairs = get_random_pairs(train_df, iteration + 1)

    # Prepare synthesis comparison with best vs current
    original_traces = source_test_df.iloc[:, 2:].values.tolist()
    synthesis_comparison = ""
    for i, (orig, best_synth, curr_synth, best_err, curr_err) in enumerate(
            zip(original_traces, best_synthesized, current_synthesized, best_errors, current_errors)):
        synthesis_comparison += f" ({orig}, {best_synth}, {curr_synth}), Error: {int(curr_err)}\n\n"

    # Create prompt and get response
    prompt = create_optimizer_prompt(sample_pairs, synthesis_comparison)
    new_synthesized = get_ai_response_with_retry(
        prompt, f"(iteration {iteration + 1})")

    # Calculate new errors
    new_errors = []
    source_websites = source_test_df['Website'].values
    for trace, website_id in zip(new_synthesized, source_websites):
        error = compute_mse(train_df, trace, website_id)
        new_errors.append(error)

    mean_error = np.mean(new_errors)
    prev_mean_error = np.mean(current_errors)
    best_mean_error = np.mean(best_errors)
    improvement = prev_mean_error - mean_error

    print(f"Mean error: {mean_error:.4f} (improvement: {improvement:+.4f})")
    print(f"Best mean error so far: {best_mean_error:.4f}")

    return new_synthesized, new_errors


def plot_results(errors_history, synthesized_history, source_test_df, train_df):
    """Plot error progression and final trace comparison"""
    # Plot error progression
    plt.figure(figsize=(12, 6))
    errors_array = np.array(errors_history)

    # Individual trace errors
    for i in range(errors_array.shape[1]):
        plt.plot(errors_array[:, i], alpha=0.6, label=f'Trace {i+1}')

    # Mean error
    mean_errors = np.mean(errors_array, axis=1)
    plt.plot(mean_errors, 'k-', linewidth=3, label='Mean Error')

    plt.xlabel('Iteration')
    plt.ylabel('MSE Error')
    plt.title('Error Progression During Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot final trace comparison
    original_traces = source_test_df.iloc[:, 2:].values
    final_synthesized = synthesized_history[-1]

    # Get target traces
    source_websites = source_test_df['Website'].values
    target_traces = []
    for web_id in source_websites:
        candidates = train_df[(train_df['Website'] == web_id) &
                              (train_df['Location'] == CONFIG['target_location'])]
        if len(candidates) > 0:
            target_trace = candidates.iloc[:, 2:].median(axis=0).values
            target_traces.append(target_trace)
        else:
            target_traces.append(np.zeros(CONFIG['data_points']))

    n_traces = len(original_traces)
    fig, axes = plt.subplots(n_traces, 1, figsize=(15, 4 * n_traces))
    if n_traces == 1:
        axes = [axes]

    for i, (orig, synth, target) in enumerate(zip(original_traces, final_synthesized, target_traces)):
        axes[i].plot(orig, label='Original (Source)', alpha=0.7, linewidth=2)
        axes[i].plot(synth, label='Synthesized', alpha=0.7, linewidth=2)
        axes[i].plot(target, label='Target', alpha=0.7, linewidth=2)
        axes[i].set_title(f'Trace {i+1} - Final Result')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Packet Count')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('final_trace_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return mean_errors


def save_results(errors_history, synthesized_history):
    """Save results to files"""
    os.makedirs("results", exist_ok=True)

    # Save error history
    errors_df = pd.DataFrame(errors_history)
    errors_df.to_csv("results/error_history.csv", index=False)

    # Save synthesized traces
    import pickle
    with open("results/synthesized_history.pkl", 'wb') as f:
        pickle.dump(synthesized_history, f)

    # Save config
    import json
    with open("results/config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print("Results saved to results/")


def main():
    """Main execution function"""
    print("Starting DoH Trace Synthesis and Optimization...")
    print(f"Config: {CONFIG['max_iterations']} iterations, "
          f"{CONFIG['source_location']} -> {CONFIG['target_location']}")
    print(f"Data points per trace: {CONFIG['data_points']}")

    # Initialize API clients
    initialize_clients()

    # Load data
    train_df, test_df = load_data()

    # Initial synthesis
    current_synthesized, current_errors, source_test_df = initial_synthesis(
        train_df)

    # Store history
    errors_history = [current_errors]
    synthesized_history = [current_synthesized]

    # Track best synthesis
    best_synthesized = current_synthesized.copy()
    best_errors = current_errors.copy()
    best_mean_error = np.mean(current_errors)

    # Optimization loop
    for iteration in range(CONFIG['max_iterations']):
        try:
            current_synthesized, current_errors = optimize_iteration(
                train_df, source_test_df, current_synthesized, current_errors,
                iteration, best_synthesized, best_errors
            )

            errors_history.append(current_errors)
            synthesized_history.append(current_synthesized)

            # Update best synthesis if current is better
            current_mean_error = np.mean(current_errors)
            if current_mean_error < best_mean_error:
                best_synthesized = current_synthesized.copy()
                best_errors = current_errors.copy()
                best_mean_error = current_mean_error
                print(
                    f"🎉 New best mean error: {best_mean_error:.4f} (iteration {iteration + 1})")

            # Progress update every 10 iterations
            if (iteration + 1) % 10 == 0:
                mean_error = np.mean(current_errors)
                print(f"\nProgress: {iteration + 1}/{CONFIG['max_iterations']} "
                      f"iterations completed, Current mean error: {mean_error:.4f}")
                print(f"Best mean error: {best_mean_error:.4f}")
                print(
                    f"API key usage distribution: {[(i+1, (iteration+1) // len(clients) + (1 if i < (iteration+1) % len(clients) else 0)) for i in range(len(clients))]}")

        except Exception as e:
            print(f"Error at iteration {iteration + 1}: {e}")
            continue

    print("\n=== OPTIMIZATION COMPLETE ===")

    # Generate visualizations
    print("Generating visualizations...")
    mean_errors = plot_results(
        errors_history, synthesized_history, source_test_df, train_df)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Initial mean error: {np.mean(errors_history[0]):.4f}")
    print(f"Final mean error: {np.mean(errors_history[-1]):.4f}")
    print(
        f"Total improvement: {np.mean(errors_history[0]) - np.mean(errors_history[-1]):.4f}")
    print(f"Best iteration: {np.argmin(mean_errors) + 1}")
    print(f"Best mean error: {np.min(mean_errors):.4f}")
    print(
        f"Total API calls made: {len(errors_history) * CONFIG['max_retries']}")
    print(f"API keys used: {len(clients)}")

    # Save results
    save_results(errors_history, synthesized_history)

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
