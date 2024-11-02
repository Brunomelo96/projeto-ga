import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_metrics(train_loss, test_loss, train_acc, test_acc, precision, recall, f1, filepath):
    plt.figure(figsize=(15, 7))

    loss_and_acc_path = '_loss_and_accuracy.png'
    # metrics_path = '_metrics.png'
    epochs_array = np.arange(0, len(train_loss))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_array, train_loss, label='train_loss')
    plt.plot(epochs_array, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_array, train_acc, label='train_accuracy')
    plt.plot(epochs_array, test_acc, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.savefig(f'{filepath}{loss_and_acc_path}')
    plt.close()

    # plt.plot(epochs_array, precision, label='precision')
    # plt.plot(epochs_array, recall, label='recall')
    # plt.plot(epochs_array, f1, label='f1')

    # plt.title('Metrics')
    # plt.xlabel('Epochs')
    # plt.legend()

    # plt.savefig(f'{filepath}_scores.png')
    # plt.close()


def plot_confusion_matrix(model, test_loader, filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_pred = []
    y_labels = []

    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            outputs = torch.softmax(outputs, dim=1)

            predicted = outputs.argmax(dim=1)

            y_pred.extend(predicted.cpu().numpy())
            y_labels.extend(labels.cpu().numpy())

    plt.figure(figsize=(15, 7))

    cm = confusion_matrix(y_labels, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.savefig(filepath)
    plt.close()


def plot_probabilities(probabilities_dict, filepath):
    entries = probabilities_dict.items()
    probabilities_path = '_probabilities'
    for key, values in entries:
        plt.title('Probabilidades')
        plt.xlabel('Probabilidade para 0')
        plt.ylabel('Probabilidade para 1')
        x = [value[0] for value in values]
        print(x, "x")
        y = [value[1] for value in values]
        plt.scatter(x, y, label=f'Epoch {key}')

        plt.savefig(f'{filepath}{probabilities_path}_{key}.png')
        plt.close()


def plot_grayscale_histogram(base_path, output_path):
    plot_hsv_histogram(f'{base_path}', 'grayscale')
    plot_hsv_histogram(f'{base_path}', 'rgb')


def plot_hsv_histogram(base_path, filepath):
    plt.title(f'Hues - {filepath}')

    entries_path = f'{base_path}/{filepath}'
    for entry in os.scandir(entries_path):
        print(entry.path, "path")
        img = cv2.imread(entry.path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        hue_histogram = cv2.calcHist([h], [0], None, [180], [0, 180])
        plt.plot(hue_histogram)

    plt.savefig(f'./{filepath}-hue.png')
    plt.close()

    plt.title('Saturation')

    for entry in os.scandir(entries_path):
        print(entry.path, "path")
        img = cv2.imread(entry.path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        s_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
        plt.plot(s_hist)

    plt.savefig(f'./{filepath}-saturation.png')
    plt.close()
