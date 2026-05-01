import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(8,5))
    plt.plot(history['loss_train'], label='Train Loss', color='blue')
    plt.plot(history['loss_test'], label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.show()