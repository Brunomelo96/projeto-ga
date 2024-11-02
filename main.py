from runners.default import run_net
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ga.default import run
# from writers.tar_writer import write_images
# from writers.zip_writer import write_images

dataset_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

base_path = '/content/gdrive/MyDrive/datasets/Candidatos/GA'

train = datasets.ImageFolder(
    root=f'{base_path}/train',
    # root="./train",
    transform=dataset_transform
)
# validation dataset
test = datasets.ImageFolder(
    root=f'{base_path}/test',
    # root="./test",
    transform=dataset_transform
)

# run_net([16, 20, 14, 5, 2, 4], train, test,  4)
# tar writer
# write_images('./all.zip', './train.tar', './test.tar', '0.2', None)
# zip_writer
# write_images('./original_dataset', './train', './test')
# best = run('/content/gdrive/MyDrive/datasets/Candidatos/GA')
best = run(train, test)
print(best, "best")
