from backend.model.model_wrapper import ModelWrapper
from dotenv import load_dotenv
import os
import argparse
from backend.cli import run_cli

load_dotenv()

MAX_LENGTH_SENT = int(os.getenv('MODEL_MAX_LENGTH_SENT'))
LABELS_PATH = os.getenv('LABELS_PATH')
MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH')
MODEL_WEIGHTS_PATH = os.getenv('MODEL_WEIGHTS_PATH')
VOCAB_PATH = os.getenv('VOCAB_PATH')
BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE')) 

model_wrapper = ModelWrapper(
    max_length_sent=MAX_LENGTH_SENT,
    labels_path=LABELS_PATH,
    model_config_path=MODEL_CONFIG_PATH,
    model_weights_path=MODEL_WEIGHTS_PATH,
    vocab_path=VOCAB_PATH,
    batch_size=BATCH_SIZE
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['cli', 'server'], default='cli', help='Run type')
    args = parser.parse_args()

    if args.type == 'cli':
        run_cli(model_wrapper)
    else:
        pass