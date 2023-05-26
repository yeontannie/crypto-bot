import seaborn as sns
import matplotlib.pyplot as plt

def plot_anomalies(data, anomalies, crypto):
    plt.plot(data.index, data['close'], label='Close price', lw=1.5)
    plt.scatter(x=anomalies.index, y=anomalies['close'], label='Anomaly', color='red', s=2.5, zorder=2)
    plt.title(f'Detected anomalies in {crypto}')
    plt.legend()
    plt.show()

def plot_threshold(data, label):
    plt.plot(data.index, data['loss'], label=label)
    plt.plot(data.index, data['threshold'], label='Threshold')
    plt.legend()
    plt.title(f'{label} vs. Threashold')
    plt.show()

def build_sns_distplot(data, label):
    sns.distplot(data, bins=50, kde=True, label=label)
    plt.legend()
    plt.show()

def plot_loss(loss):
    epochs = range(len(loss))

    plt.plot(epochs, loss, label='model 1 loss')
    #plt.plot(epochs, losses[1], label='model 2 loss')
    #plt.plot(epochs, losses[2], label='model 3 loss')
    #plt.plot(epochs, losses[3], label='model 4 loss')
    #plt.plot(epochs, losses[4], label='model 5 loss')
    plt.title('Training loss')
    plt.legend(loc=0)
    plt.show()

def plot_predictions(test_pred_df, original_df):
    plt.plot(original_df['date'], original_df['close'], color="royalblue", label='Actual price')
    plt.plot(test_pred_df['date'], test_pred_df['close'], color="red", label='Predicted data', zorder=2)

    plt.title("Original vs Predicted")
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()
