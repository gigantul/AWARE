# config/config.py

# Device maps for model parallelism
opt_30b_device_map = {
    **{f'model.decoder.layers.{i}': 0 for i in range(25)},
    **{f'model.decoder.layers.{i}': 1 for i in range(25, 49)},
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.final_layer_norm': 1,
    'lm_head': 1,
}

opt_13b_device_map = {
    **{f'model.decoder.layers.{i}': 0 for i in range(22)},
    **{f'model.decoder.layers.{i}': 1 for i in range(22, 40)},
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.final_layer_norm': 1,
    'lm_head': 1,
}

# Default model config
MODEL_NAME = "facebook/opt-13b"
DEVICE_MAP = opt_13b_device_map

# File paths
DATA_DIR = "data"
DATASET_PATH = f"{DATA_DIR}/sampleQA.json"
OUTPUT_DIR = "results"
HF_DATASETS_CACHE = "hf_cache"