import json
import os
import pickle


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_pickle(obj, out_dir, filename):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, out_dir, filename):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path
