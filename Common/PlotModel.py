import matplotlib.pyplot as plt


def plot_graphs(model_history, metric):
    metric = str(metric).lower()
    print(metric)
    print(len(model_history.history[metric]))
    tr_label = "Training " + metric.title()
    print(tr_label)
    plt.plot(model_history.history[metric], 'r', label=tr_label)
    val_label = "Validation " + metric.title()
    val_key = 'val_' + metric
    and_validation = ''
    if model_history.history.get(val_key) is not None:
        plt.plot(model_history.history['val_' + metric], 'b', label=val_label)
        and_validation = ' and validation'
    plt.xlabel('Epochs')
    plt.ylabel(metric.title())
    plt.legend()
    plt.title(f'Training{and_validation} {metric}')
    plt.figure()
    plt.show()
