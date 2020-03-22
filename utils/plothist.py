import pickle
import matplotlib.pyplot as plt

def plot_history(histpath):
    with open(histpath,'rb') as file_pi:
        hist1=pickle.load(file_pi)

    for key in ['loss', 'val_loss']:
        plt.plot(hist1[key],label=key)
    plt.legend()
    plt.savefig('loss.png')
