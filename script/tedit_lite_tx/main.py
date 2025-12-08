import os
import sys
import argparse
from pathlib import Path


def main(args):
    # Paths - use __file__ to be robust to working directory
    ROOT = Path(__file__).resolve().parent.parent.parent  # Go up from script/tedit_lite_tx/main.py to project root
    tedit_path = ROOT / "script" / "tedit_lite_tx"
    vital_path = ROOT / "script" / "VITAL"
    
    # Convert to strings for compatibility
    tedit_path = str(tedit_path)
    vital_path = str(vital_path)
    
    # Add paths to sys.path so imports work correctly
    if tedit_path not in sys.path:
        sys.path.insert(0, tedit_path)
    if vital_path not in sys.path:
        sys.path.insert(0, vital_path)

    # Preserve original working directory
    original_cwd = os.getcwd()

    # Notebook-level configuration
    dataset_name = args.dataset_name
    tedit_mdl = args.tedit_mdl
    attr_suffix = ""
    vital_suffix = ""
    if args.open_vocab:
        vital_suffix = "_open"
    # Make these visible to the executed scripts
    global_vars = dict(
        tedit_path=tedit_path,
        vital_path=vital_path,
        dataset_name=dataset_name,
        tedit_mdl=tedit_mdl,
        attr_suffix=attr_suffix,
        vital_suffix=vital_suffix,
        open_vocab=args.open_vocab,
        __name__="__main__",
        __file__=os.path.join(tedit_path, "tedit_runs/prepare_dfs.py"),
        sys=sys,
        os=os
    )

    try:
        # 1) Prepare dataframes (from Cell 1)
        # Change to tedit_path first (prepare_dfs.py will change to vital_path internally)
        os.chdir(tedit_path)
        with open(os.path.join(tedit_path, "tedit_runs/prepare_dfs.py")) as f:
            exec(f.read(), global_vars)
        
        # Restore working directory after prepare_dfs.py (it changes it internally)
        os.chdir(tedit_path)

        # 2) Train (from Cell 2)
        global_vars.update(dict(resume=False, train=True))
        with open(os.path.join(tedit_path, "tedit_runs/train.py"), "r") as file:
            exec(file.read(), global_vars)

        # 3) Evaluate (from Cell 3)
        with open(os.path.join(tedit_path, "tedit_runs/eval.py")) as f:
            exec(f.read(), global_vars)

        model = global_vars.get("model", None)
        if model is not None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Total model parameters: {n_params}")
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--tedit_mdl", required=True)
    parser.add_argument("--open_vocab", action="store_true")
    args = parser.parse_args()
    main(args)


