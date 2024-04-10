from pathlib import Path

def get_config():
    return {
        'batch_size': 1,
        'num_epochs': 1,
        'lr': 10**-4,
        'embed_size': 6,
        'ff_intermediate_size': 32,
        'num_encoderblocks': 1,
        'num_decoderblocks': 1,
        'num_attention_heads': 2,
        'src_language': 'en',
        'tgt_language': 'fr',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
        'dropout_prob': 0.1,
        'edu': False
    }

def get_model_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)
