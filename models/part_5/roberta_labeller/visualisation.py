import matplotlib.pyplot as plt

def plot_one_curve(data, title="Training Loss", ylabel="Loss"):
    """Plot one loss curve over time

    Args:
        data (list): The loss over time (epochs, batches, samples)
        title (str, optional): Title of the diagram. Defaults to "Training Loss".
        ylabel (str, optional): Label of the y axis. Defaults to "Loss".
    """
    plt.figure(figsize=(8, 5))
    plt.plot(data, label=ylabel)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_two_curves(first, second, title="Training Metrics", first_label="Loss", second_label="Accuracy", save_path=None):
    """Plot the loss and accuracy in one diagram.

    The blue curve is the training loss and the red curve is the accuracy scores.

    Args:
        losses (list): List of loss values per epoch
        acc_scores (list): List of accuracy scores per epoch
        title (str, optional): Title of the diagram. Defaults to "Training Metrics".
        save_path (str, optional): If provided, saves the figure to this path instead of showing it.
    """
    epochs = range(1, len(first) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # First curve
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(first_label, color='tab:blue')
    ax1.plot(epochs, first, color='tab:blue', label=first_label)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Second curve (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel(second_label, color='tab:purple')
    ax2.plot(epochs, second, color='tab:purple', label=second_label)
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
