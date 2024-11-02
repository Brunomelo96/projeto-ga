from runners.default import run_net
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ga.default import run
# from writers.tar_writer import write_images
# from writers.zip_writer import write_images

dataset_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

base_path = '/content/gdrive/MyDrive/datasets/GA'
# base_path = '.'

train = datasets.ImageFolder(
    root=f'{base_path}/train',
    transform=dataset_transform
)
# validation dataset
test = datasets.ImageFolder(
    root=f'{base_path}/test',
    transform=dataset_transform
)

# run_net([16, 20, 14, 5, 2, 4], train, test,  4)
best = run(train, test, base_path)
print(best, "best")
