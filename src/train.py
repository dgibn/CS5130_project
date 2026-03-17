from dataloader import load_data
from Q_learning import Q_learning
from config import *
import pickle

def train():
    ticker = "GOOG"
    period = "10y"
    train_df, test_df, n_states, scaler = load_data(ticker, TRAIN_SIZE, period)
    Q_table = Q_learning(train_df, DAYS_WINDOW, ACTIONS, ACTIONS1, NUM_EPISODES, GAMMA, EPSILON, DECAY_RATE)
    return Q_table

if __name__ == "__main__":
    Q_table = train()
    
    # Save the Q-table dict to a file
    with open('Q_table.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)