import argparse
from pathlib import Path

def run(args):
    print(f"overwrite     : {args.overwrite}")
    print(f"dataset_name  : {args.dataset_name}")
    print(f"attr_suffix   : {args.attr_suffix}")
    print(f"suffix        : {args.suffix}")
    print(f"open_vocab    : {args.open_vocab}")
    print(f"alpha_init    : {args.alpha_init}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")       
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--attr_suffix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--open_vocab", action="store_true") # Boolean, --open_vocab -> True   |  (absent) -> False
    parser.add_argument("--alpha_init", type=float, default=None, metavar="FLOAT")

    args = parser.parse_args()
    if args.open_vocab:
        args.suffix = "_open" # results store in results/subfolder = "dataset_name"+"_"+"attr_suffix"+"_"+"suffix"
    run(args)
    # disable all plotting
    import os, importlib
    os.environ["MPLBACKEND"] = "Agg"              # headless backend
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
   
    
    # project directory (root of main)
    ROOT = Path(__file__).resolve().parent      
    RUN  = ROOT / "run"
    # make sure a results/ folder exists 
    RESULTS = ROOT / "results"
    RESULTS.mkdir(exist_ok=True)    

    globals()["args"] = args
    globals().update(vars(args)) 

    # Step-by-step execution in *one* shared namespace
    for script in ["settings.py", "data.py", "model.py", "train_2steps.py"]:
        exec((RUN / script).read_text(), globals())
    # Evaluation
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        globals()["w"] = w                         
        exec((RUN / "eval.py").read_text(),     globals())
        exec((RUN / "eng_eval.py").read_text(), globals())
    
    
