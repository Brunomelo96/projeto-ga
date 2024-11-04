import torch.optim as optim
import torch
import torch.nn as nn
from helpers.metrics import get_metrics
from torcheval.metrics.functional import multiclass_f1_score, multiclass_recall, multiclass_precision, multiclass_accuracy
from helpers.plotters import plot_metrics, plot_confusion_matrix


def train(model, trainloader, testloader, name, num_epochs=4, base_path='.'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.)
    print(device, 'device')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.BCELoss()

    # optimizer = torch.optim.RMSprop(
    #     model.parameters(),
    #     lr=0.001,
    #     momentum=0.9,
    #     weight_decay=0.9
    # )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.94)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    recalls = []
    f1s = []
    precisions = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        train_count = 0
        test_count = 0

        current_train_acc = 0
        current_test_acc = 0

        model.train()
        running_loss = 0.0
        train_iter = iter(trainloader)
        while data := next(train_iter, None):
            train_count += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            outputs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted ==
                   labels).sum().item()/len(predicted)
            current_train_acc += acc

            # print statistics
            running_loss += loss.item()
            # if train_count % 2000 == 1999:    # print every 2000 mini-batches
            #     print(
            #         f'[{epoch + 1}, {train_count + 1:5d}] loss: {running_loss / train_count:.3f}')
            print(
                f'Train: [{epoch + 1}, {train_count + 1:5d}] loss: {running_loss / train_count:.3f} acc: {acc}')

        train_loss.append(running_loss/train_count)
        train_acc.append(current_train_acc/train_count)

        # print(f'Train ACC: {current_train_acc/train_count}')

        print('Finished Training')

        running_test_loss = 0

        y_pred = []
        y_true = []

        with torch.no_grad():
            model.train(False)
            test_iter = iter(testloader)
            while data := next(test_iter, None):
                test_count += 1
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                outputs = torch.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, 1)
                current_test_acc += (predicted ==
                                     labels).sum().item()/len(predicted)

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                print(
                    f'Test: [{epoch + 1}, {test_count + 1:5d}] loss: {running_test_loss / test_count:.3f}')

        test_loss.append(running_test_loss/test_count)
        test_acc.append(current_test_acc/test_count)

        # print(f'TEste ACC: {current_test_acc/test_count}')

        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)
        f1 = multiclass_f1_score(y_pred_tensor, y_true_tensor, num_classes=90)
        recall = multiclass_recall(y_pred_tensor, y_true_tensor, num_classes=90)
        precision = multiclass_precision(y_pred_tensor, y_true_tensor, num_classes=90)
        # acc = multiclass_accuracy(y_pred_tensor, y_true_tensor, num_classes=90)
        f1s.append(f1.cpu().numpy())
        recalls.append(recall.cpu().numpy())
        precisions.append(precision.cpu().numpy())
        print(f1, recall, precision, "scores")
        # print(y_true_tensor, "y_true_tensor")
        # print(y_pred_tensor, "y_pred_tensor")
        # precision, recall, f1 = get_metrics(y_true_tensor, y_pred_tensor)

        # precisions.append(precision)
        # recalls.append(recall)
        # f1s.append(f1)
        # if epoch % 2 == 0:
        #     scheduler.step()
    plot_metrics(train_loss, test_loss, train_acc,
                 test_acc, precisions, recalls, f1s, f'{base_path}/{name}')
    plot_confusion_matrix(model, testloader, f'{base_path}/{name}')

    return test_acc[len(test_acc) - 1]
