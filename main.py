# from runners.default import run_net
from ga.default import run
# from writers.tar_writer import write_images
# from writers.zip_writer import write_images

# run_net([16, 20, 14, 5, 2, 4], 64)
# tar writer
# write_images('./all.zip', './train.tar', './test.tar', '0.2', None)
# zip_writer
# write_images('./original_dataset', './train', './test')
best = run('/content/gdrive/MyDrive/datasets/Candidatos/GA')
print(best, "best")
