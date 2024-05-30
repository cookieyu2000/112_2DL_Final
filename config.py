# config.py:
class Config:
    SEED = 42
    VAL_SPLIT = 0.3
    TEST_SPLIT = 0.5
    IMAGE_SIZE = [512, 512]
    BATCH_SIZE = 32
    EPOCHS = 100
    VERBOSE = 1
    CLASSES = ['0', '1', '2', '3', '4']
    LR = 1e-4
    PATIENCE = 20
    DECAY_FACTOR = 0.1
    DEVICE = 0
    #AUTOTUNE = tf.data.AUTOTUNE

config = Config()
