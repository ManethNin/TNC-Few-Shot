"""
Simple replacement for TimeSynth that works on Apple Silicon M4
This creates the same simulated data without the problematic TimeSynth dependency
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import pickle

n_signals = 5
n_states = 4
transition_matrix = np.eye(n_states)*0.85
transition_matrix[0,1] = transition_matrix[1,0] = 0.05
transition_matrix[0,2] = transition_matrix[2,0] = 0.05
transition_matrix[0,3] = transition_matrix[3,0] = 0.05
transition_matrix[2,3] = transition_matrix[3,2] = 0.05
transition_matrix[2,1] = transition_matrix[1,2] = 0.05
transition_matrix[3,1] = transition_matrix[1,3] = 0.05


def simple_signal_generator(state, window_size):
    """Simple signal generator that replaces TimeSynth"""
    np.random.seed(state * 42)  # For reproducibility
    
    if state == 0:
        # Periodic signal
        t = np.linspace(0, 4*np.pi, window_size)
        signal = np.sin(t) + 0.5*np.sin(3*t) + np.random.normal(0, 0.3, window_size)
    elif state == 1:
        # NARMA-like autoregressive signal
        signal = np.zeros(window_size)
        signal[0] = np.random.normal(0, 0.5)
        for i in range(1, window_size):
            if i >= 5:
                signal[i] = 0.3*signal[i-1] + 0.05*signal[i-5] + 0.1*signal[i-1]*signal[i-5] + np.random.normal(0, 0.3)
            else:
                signal[i] = 0.3*signal[i-1] + np.random.normal(0, 0.3)
    elif state == 2:
        # Smooth Gaussian process-like signal
        signal = np.random.normal(0, 1, window_size)
        # Apply smoothing
        for i in range(1, window_size):
            signal[i] = 0.8*signal[i-1] + 0.2*signal[i]
        signal += np.random.normal(0, 0.1, window_size)
    elif state == 3:
        # Another autoregressive pattern
        signal = np.zeros(window_size)
        signal[0] = np.random.normal(0, 0.5)
        for i in range(1, window_size):
            if i >= 3:
                signal[i] = 0.1*signal[i-1] + 0.25*signal[i-2] + 2.5*signal[i-3] - 0.005*signal[i-1]*signal[i-2]*signal[i-3] + np.random.normal(0, 0.3)
            else:
                signal[i] = 0.3*signal[i-1] + np.random.normal(0, 0.3)
    
    return signal


def create_signal(sig_len, window_size=50):
    states = []
    sig_1 = []
    sig_2 = []
    sig_3 = []
    pi = np.ones((1,n_states))/n_states

    for _ in range(sig_len//window_size):
        current_state = np.random.choice(n_states, 1, p=pi.reshape(-1))
        states.extend(list(current_state)*window_size)

        current_signal = simple_signal_generator(current_state[0], window_size)
        sig_1.extend(current_signal)
        correlated_signal = current_signal*0.9 + .03 + np.random.randn(len(current_signal))*0.4
        sig_2.extend(correlated_signal)
        uncorrelated_signal = simple_signal_generator((current_state[0]+2)%4, window_size)
        sig_3.extend(uncorrelated_signal)

        pi = transition_matrix[current_state]
    signals = np.stack([sig_1, sig_2, sig_3])
    return signals, states


def normalize(train_data, test_data, config='mean_normalized'):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    sig_len = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == 'mean_normalized':
        feature_means = np.mean(train_data, axis=(0,2))
        feature_std = np.std(train_data, axis=(0, 2))
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = train_data - feature_means[np.newaxis,:,np.newaxis]/\
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis,:,np.newaxis]
        test_data_n = test_data - feature_means[np.newaxis, :, np.newaxis] / \
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


def main(n_samples, sig_len):
    print("Creating simulated dataset without TimeSynth (M4 compatible)...")
    all_signals = []
    all_states = []
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"Generated {i}/{n_samples} samples...")
        sample_signal, sample_state = create_signal(sig_len)
        all_signals.append(sample_signal)
        all_states.append(sample_state)

    dataset = np.array(all_signals)
    states = np.array(all_states)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    train_data_n, test_data_n = normalize(train_data, test_data)
    train_state = states[:n_train]
    test_state = states[n_train:]

    print("Dataset Shape ====> \tTrainset: ", train_data_n.shape, "\tTestset: ", test_data_n.shape)

    ## Save signals to file
    if not os.path.exists('./data/simulated_data'):
        os.mkdir('./data/simulated_data')
    with open('./data/simulated_data/x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open('./data/simulated_data/x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open('./data/simulated_data/state_train.pkl', 'wb') as f:
        pickle.dump(train_state, f)
    with open('./data/simulated_data/state_test.pkl', 'wb') as f:
        pickle.dump(test_state, f)
    
    print("✅ Simulated data created successfully!")
    print("Files saved in ./data/simulated_data/")


if __name__ == '__main__':
    main(n_samples=500, sig_len=2000)
