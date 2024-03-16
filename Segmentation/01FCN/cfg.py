BATCH_SIZE = 4
EPOCH_NUMBER = 1
DATASET = ['CamVid', 12]

crop_size = (352, 480)

class_dict_path = './Datasets/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = './Datasets/' + DATASET[0] + '/train'
TRAIN_LABEL = './Datasets/' + DATASET[0] + '/train_labels'
VAL_ROOT = './Datasets/' + DATASET[0] + '/val'
VAL_LABEL = './Datasets/' + DATASET[0] + '/val_labels'

TEST_ROOT = './Datasets/' + DATASET[0] + '/test'
TEST_LABEL = './Datasets/' + DATASET[0] + '/test_labels'



