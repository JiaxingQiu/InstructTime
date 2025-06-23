import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml

from tedit_data import *
from tedit_model import *


# _, meta = TEditDataset(df_train, df_test, df_left, config_dict, split="train").get_loader(batch_size=128)
def load_model(meta, dataset_name, mdl_name):
    # config
    configs = yaml.safe_load(open("tedit_configs/"+mdl_name+"/"+dataset_name+".yaml")) 
    configs['model']['attrs']['num_attr_ops'] = meta['attr_n_ops']
    configs['train']['output_folder'] = "tedit_save/"+mdl_name+"/"+dataset_name

    # load checkpoint 
    model = ConditionalGenerator(configs['model'])
    return model, configs


# Define a custom dataset returning the full dict per sample
class DictDataset(torch.utils.data.Dataset):
    def __init__(self, src_x, tgt_x, src_attrs, tgt_attrs, tp):
        self.src_x = src_x
        self.tgt_x = tgt_x
        self.src_attrs = src_attrs
        self.tgt_attrs = tgt_attrs
        self.tp = tp

    def __len__(self):
        return self.src_x.shape[0]

    def __getitem__(self, idx):
        return {
            "src_x": self.src_x[idx],
            "tgt_x": self.tgt_x[idx],
            "src_attrs": self.src_attrs[idx],
            "tgt_attrs": self.tgt_attrs[idx],
            "tp": self.tp[idx],
        }

def get_eval_loader(test_loader_src, test_loader_tgt):
    # # Create source and target test loaders
    # test_loader_src, _ = TEditDataset(df_train, df_test, df_left_src, config_dict, split="test").get_loader(batch_size=128, shuffle=False)
    # test_loader_tgt, _ = TEditDataset(df_train, df_test, df_left_tgt, config_dict, split="test").get_loader(batch_size=128, shuffle=False)

    # Collect modified batches
    new_batches = []
    for batch_src, batch_tgt in zip(test_loader_src, test_loader_tgt):
        new_batches.append({
            "src_x": batch_src["x"],
            "tgt_x": torch.zeros_like(batch_src["x"]),
            "src_attrs": batch_src["attrs"],
            "tgt_attrs": batch_tgt["attrs"],
            "tp": batch_src["tp"],
        })

    # Stack each field across batches
    src_x_all     = torch.cat([b["src_x"] for b in new_batches], dim=0)
    tgt_x_all     = torch.cat([b["tgt_x"] for b in new_batches], dim=0)
    src_attrs_all = torch.cat([b["src_attrs"] for b in new_batches], dim=0)
    tgt_attrs_all = torch.cat([b["tgt_attrs"] for b in new_batches], dim=0)
    tp_all        = torch.cat([b["tp"] for b in new_batches], dim=0)

    # Create DataLoader
    final_dataset = DictDataset(src_x_all, tgt_x_all, src_attrs_all, tgt_attrs_all, tp_all)
    final_loader = DataLoader(final_dataset, batch_size=128, shuffle=False)
    return final_loader

def _plot_random_n_pairs(src, pred_tw, pred_te, n=20, rows=4, cols=5, title = '', random=False):
    import numpy as np
    """
    Plot `n` random triplets (src_x, pred_tw, pred_te) in a grid of subplots.
    """
    assert src.shape[0] >= n, f"Not enough samples: requested {n}, got {src.shape[0]}"

    if src.ndim == 3 and src.shape[-1] == 1:
        src     = src.squeeze(-1)
        pred_tw = pred_tw.squeeze(-1)
        pred_te = pred_te.squeeze(-1)

    indices = np.random.choice(src.shape[0], n, replace=False) if random else np.arange(n)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(src[idx].numpy(),     label="source ts",   linestyle="--")
        if title == "Time Editing":
            ax.plot(pred_te[idx].numpy(), label="generation | ts, attr", linestyle="-")
            ax.plot(pred_tw[idx].numpy(), label="generation | attr", linestyle=":")
        elif title == "Time Weaver":
            ax.plot(pred_tw[idx].numpy(), label="generation | attr", linestyle="-")
            
        ax.set_title(f"Sample {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6)

    for j in range(n, rows * cols):
        axes[j].axis('off')
        
    if title:
        fig.suptitle(title, fontsize=40)  # Add the figure-level title here

    plt.tight_layout()
    plt.show()



def tedit_generate(
    configs,
    test_loader,
    n_samples: int = 10,
    sampler: str = "ddpm",
    device: str = None,
    plot_n: int = None,
    pred_only = False
):
    model_ckpt = configs['train']['output_folder'] + "/ckpts/model_best.pth"
    model_cfg = configs["model"]
    if "/tw/" in configs['train']['output_folder']:  # check for "/tw/" in the path
        title = "Time Weaver"
    else:
        title = "Time Editing"
        
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load weights
    model = ConditionalGenerator(model_cfg).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    preds_cg, preds_te, src_xs, tgt_xs = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            out_cg = model.generate(batch, n_samples=n_samples, mode="cond_gen", sampler=sampler)
            out_cg = out_cg.permute(0, 1, 3, 2).median(dim=0).values

            out_te = model.generate(batch, n_samples=n_samples, mode="edit", sampler=sampler)
            out_te = out_te.permute(0, 1, 3, 2).median(dim=0).values

            preds_cg.append(out_cg.cpu())
            preds_te.append(out_te.cpu())
            src_xs.append(batch["src_x"].cpu())
            tgt_xs.append(batch["tgt_x"].cpu())

    pred_cg = torch.cat(preds_cg, dim=0)
    pred_te = torch.cat(preds_te, dim=0)
    src     = torch.cat(src_xs, dim=0)
    tgt     = torch.cat(tgt_xs, dim=0)

    if str(device).startswith("cuda"):
        del preds_cg, preds_te, src_xs, tgt_xs, out_cg, out_te, batch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if pred_only:
        return pred_te
    else: 
        if plot_n:
            _plot_random_n_pairs(src, pred_cg, pred_te, title=title, n = plot_n)
        return pred_cg, pred_te, src, tgt
    
    
def tedit_generate_ts_tx(
    df_level,
    meta,
    config_dict,
    configs,         # used by tedit_generate
    y_col,
    new_level_col_map,
):
    df_level_src = df_level.copy()
    test_loader_src, _ = TEditDataset_lite(df_level_src, meta, config_dict).get_loader(batch_size=128, shuffle=False)

    ts_hat_ls = {}
    for new_y_col in new_level_col_map.keys():
        df_level_tgt = df_level.copy()
        df_level_tgt[y_col] = new_level_col_map[new_y_col]
        test_loader_tgt, _ = TEditDataset_lite(df_level_tgt, meta, config_dict).get_loader(batch_size=128, shuffle=False)

        final_loader = get_eval_loader(test_loader_src, test_loader_tgt)
        y_hats = tedit_generate(configs, test_loader=final_loader, n_samples=1, sampler="ddpm-ddim", pred_only=True)
        y_hats = y_hats.squeeze(-1)
        ts_hat_ls[new_y_col] = list(zip(df_level[new_y_col], y_hats))

    return ts_hat_ls
