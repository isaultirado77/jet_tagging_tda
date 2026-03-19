import matplotlib.pyplot as plt


def plot_training(metrics, path):

    plt.plot(metrics['train_loss'], label='loss')
    plt.plot(metrics['val_accuracy'], label='accuracy')
    plt.plot(metrics['val_auc'], label='auc')

    plt.legend()
    plt.title('Training curves')

    plt.savefig(path)
    plt.close()

def plot_roc(fpr, tpr, out_path):
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.savefig(out_path)
    plt.close()
    