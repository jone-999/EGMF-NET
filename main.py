"""Run the task selected in conf.yml: python main.py."""
import argparse
from utils.config import DEFAULT_CONFIG, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="optional alternate config file")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config["mode"] == "train":
        from train import train
        return train(config)
    from test import run_test
    return run_test(config)


if __name__ == "__main__":
    main()
