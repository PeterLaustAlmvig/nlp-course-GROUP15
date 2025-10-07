import matplotlib.pyplot as plt

def plot_loss(losses, title="Training Loss", ylabel="Loss"):
    """Plot one loss curve over time

    Args:
        losses (list): The loss over time (epochs, batches, samples)
        title (str, optional): Title of the diagram. Defaults to "Training Loss".
        ylabel (str, optional): Label of the y axis. Defaults to "Loss".
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_and_acc(losses, acc_scores, title="Training Metrics", save_path=None):
    """Plot the loss and accuracy in one diagram.

    The blue curve is the training loss and the red curve is the accuracy scores.

    Args:
        losses (list): List of loss values per epoch
        acc_scores (list): List of accuracy scores per epoch
        title (str, optional): Title of the diagram. Defaults to "Training Metrics".
        save_path (str, optional): If provided, saves the figure to this path instead of showing it.
    """
    epochs = range(1, len(losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss curve
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(epochs, losses, color='tab:blue', label="Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Accuracy curve (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:purple')
    ax2.plot(epochs, acc_scores, color='tab:purple', label="Accuracy")
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
