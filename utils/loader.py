import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest
def get_training_data(rgb_dir, img_options, debug, dino_dir=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None, debug, dino_dir=dino_dir)

def get_validation_data(rgb_dir,  debug=False, dino_dir=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None, debug, dino_dir=dino_dir)

def get_test_data(rgb_dir,  debug=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None, debug)



