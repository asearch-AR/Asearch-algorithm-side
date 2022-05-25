import os
join = os.path.join

class PATH_ARGS:
    # path
    ROOT_DIR = os.path.dirname(__file__)
    
    # data
    ROOT_DATA_DIR = join(ROOT_DIR, 'data')
    MAIN_DATA_DIR = join(ROOT_DATA_DIR, 'data')
    MODEL_DATA_DIR = join(ROOT_DATA_DIR, 'model')

    # src
    SRC_DIR = join(ROOT_DIR, 'src')
    MODEL_DIR = join(SRC_DIR, 'model')
    FEATURE_PATH = join(SRC_DIR, 'feature')
    DATA_MANAGEMENT_PATH = join(SRC_DIR, 'data_management')

    # utils
    UTILS_DIR = join(ROOT_DIR, 'utils')

    # result
    RESULT_DIR = join(ROOT_DIR, 'result')

    # logs
    LOGS_DIR = join(ROOT_DIR, 'logs')

class Simbert_Args:
    simbert_name = 'chinese_simbert_L-12_H-768_A-12'
    dict_path = join(PATH_ARGS.MODEL_DATA_DIR, simbert_name, 'vocab.txt')
    config_path = join(PATH_ARGS.MODEL_DATA_DIR, simbert_name, 'bert_config.json')
    checkpoint_path = join(PATH_ARGS.MODEL_DATA_DIR, simbert_name, 'bert_model.ckpt')