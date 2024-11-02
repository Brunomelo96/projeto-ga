import torchvision.transforms as transforms
import webdataset as wds
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader
from trainers.default import train
from models.basic import Net

batch_size = 256


def run_net(hyperparameters, train_dataset, test_dataset, epochs=4, base_path='.'):
    print(hyperparameters, "hyperparameters")
    net = Net(
        hyperparameters[0],
        hyperparameters[1],
        hyperparameters[2],
        hyperparameters[3],
        hyperparameters[4],
        hyperparameters[5],
    )

    # net = BaseNet()

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=1,
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=1,
    )

    acc = train(net, trainloader, testloader,
                f'{hyperparameters[0]}_{hyperparameters[1]}_{hyperparameters[2]}_{hyperparameters[3]}_{hyperparameters[4]}_{hyperparameters[5]}_{epochs}', epochs, base_path)

    return acc
