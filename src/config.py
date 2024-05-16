"""""
Dataset configurations:
    :param DATASET_PATH: Directory path to dataset `.tar` files.
    :param TASK_ID: Specifies the the segmentation task ID (refer to README.md).
    :param IN_CHANNELS: Number of input channels.
    :param NUM_CLASSES: Specifies the number of output channels for disparate classes.
    :param BACKGROUND_AS_CLASS: If True, the model treats background as a class.

""" ""
DATASET_PATH = "/PATH/TO/THE/DATASET"
TASK_ID = 1
IN_CHANNELS = 1
NUM_CLASSES = 1
BACKGROUND_AS_CLASS = False


"""""
Training configurations:
    :param TRAIN_VAL_TEST_SPLIT: Delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
    :param SPLIT_SEED: Random seed with which the dataset is splitted.
    :param TRAINING_EPOCH: Number of training epochs.
    :param VAL_BATCH_SIZE: Specifies the batch size of the training `DataLoader`.
    :param TEST_BATCH_SIZE: Specifies the batch size of the test `DataLoader`.
    :param TRAIN_CUDA: If True, moves the model and inference onto GPU.
    :param BCE_WEIGHTS: The class weights for the Binary Cross Entropy loss.
""" ""
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
SPLIT_SEED = 42
TRAINING_EPOCH = 100
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
BCE_WEIGHTS = [0.004, 0.996]
