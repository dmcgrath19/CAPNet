def get_config():
    return {
        'epochs': 20,
        'batch_size': 4,
        'lr': 1e-3,
        'num_classes': 40,  # for NYU semantic seg
        'dataset_path': 'path/to/nyu_depth_v2/',
        'save_dir': './checkpoints/',
        'use_cuda': True
    }
