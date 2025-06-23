import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from config import device
import numpy as np
import pandas as pd
from tqdm import tqdm
from generation import interpolate_ts_tx
import copy
from typing import List, Tuple, Optional


# Add the tedit_lite and tedit_lite_tx folders to the system path
import sys, os
tedit_attr_path = os.path.abspath("../tedit_lite")
if tedit_attr_path not in sys.path:
    sys.path.append(tedit_attr_path)
from tedit_generation import tedit_generate_ts_tx as tedit_generate_ts_tx
tedit_tx_path = os.path.abspath("../tedit_lite_tx")
if tedit_tx_path not in sys.path:
    sys.path.append(tedit_tx_path)
from tedit_tx_generation import tedit_generate_ts_tx as tedit_tx_generate_ts_tx


class TSClassifier(nn.Module):
    """Time‑series encoder + linear head"""

    def __init__(self, 
                 vital_model: nn.Module, 
                 df: pd.DataFrame,  # df_left
                 config_dict: dict,
                 y_col: str,
                 finetune: bool = True):
        super().__init__()
        self.y_col = y_col
        self.y_levels = list(df[y_col].unique())
        self.n_classes = len(self.y_levels)
        self.ts_cols = [str(i + 1) for i in range(config_dict["seq_length"])]
        self.encoder = copy.deepcopy(vital_model.ts_encoder).to(device).eval()
        if not finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.LazyLinear(self.n_classes)  # infers in‑features on 1st call

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return *logits* of shape (B, n_classes)."""
        _, mean, *_ = self.encoder(x)  # (z, mean, ...)
        return self.head(mean)
    
    def _stratified_bootstrap(self, group, b):
        n = len(group)
        replace = n < b
        return group.sample(n=b, replace=replace, random_state=333)
    
    def _prepare_df2train(self, df: pd.DataFrame, config_dict: dict, conditions: Optional[List[Tuple[str, str]]] = None, b: Optional[int] = None):
        df_work = df.copy()
        if conditions is not None:
            for col, val in conditions:
                df_work = df_work[df_work[col] == val]
        if b is not None:
            df2train = (
                df_work.groupby(self.y_col, group_keys=False)
                .apply(self._stratified_bootstrap, b=b)
                .reset_index(drop=True)
            )
        else:
            df2train = df_work.copy()
        if not config_dict['open_vocab']:
            df2train = df2train[
                self.ts_cols + config_dict["txt2ts_y_cols"] + ["text", "label"]
            ].copy()
        else:
            df2train = df2train[
                self.ts_cols + config_dict["txt2ts_y_cols"] + 
                [col+'_aug' for col in config_dict["txt2ts_y_cols"]] + 
                ["text", "label"]
            ].copy()


        if config_dict.get("ts_global_normalize", False):
            global_mean = config_dict["ts_normalize_mean"]
            global_std = config_dict["ts_normalize_std"]
            df2train[self.ts_cols] = (df2train[self.ts_cols] - global_mean) / global_std

        return df2train
    
    def fit(
        self,
        df_train,
        *,
        epochs: int = 10_000,
        batch_size: int = 256,
        lr: float = 1e-5,
        patience: int = 20,
        min_delta: float = 0.0,
        plot: bool = False,
        device: str | torch.device = "cpu",
    ) -> Tuple[List[float], List[float]]:
        """Fit *in‑place* with early stopping. Returns (train_losses, val_losses)."""

        self.to(device)
        self.train()
        label2idx = {lvl: i for i, lvl in enumerate(self.y_levels)} # # label ↔ idx map

        X_all = torch.tensor(df_train[self.ts_cols].values, dtype=torch.float32).to(device)
        y_all = torch.tensor(
            df_train[self.y_col].map(label2idx).values, dtype=torch.long
        ).to(device)
        ds_full = TensorDataset(X_all, y_all)

        train_len = int(0.8 * len(ds_full))
        val_len = len(ds_full) - train_len
        train_ds, val_ds = random_split(
            ds_full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        train_losses, val_losses = [], []
        best_val = float("inf")
        stale = 0
        interval = max(1, epochs // 5)
        best_state = None

        for ep in tqdm(range(epochs), desc="clf‑train", leave=False):
            # ---- train ----
            run = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()
                run += loss.item() * xb.size(0)
            train_loss = run / train_len
            train_losses.append(train_loss)

            # ---- val ----
            self.eval(); run = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    run += criterion(self(xb), yb).item() * xb.size(0)
            val_loss = run / val_len
            val_losses.append(val_loss)
            self.train()

            # ---- early stopping ----
            if val_loss < best_val - min_delta:
                best_val = val_loss
                stale = 0
                best_state = {k: v.cpu() for k, v in self.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {ep + 1}")
                    if best_state is not None:
                        self.load_state_dict(best_state)
                    break

            if plot and ((ep + 1) % interval == 0 or ep == 0):
                print(
                    f"epoch {ep + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f}"
                )

        if plot:
            plt.plot(train_losses, label="train")
            plt.plot(val_losses, label="val")
            plt.xlabel("epoch"); plt.ylabel("loss")
            plt.title("Training vs. validation loss")
            plt.grid(True); plt.legend(); plt.show()

        # return train_losses, val_losses

    @torch.no_grad()
    def predict(
        self,
        df_pred: pd.DataFrame,
        device: str | torch.device = "cpu",
    ) -> pd.DataFrame:
        """Append soft‑max probs to `df_pred` and return the modified frame."""
        self.to(device)
        self.eval()
        X = torch.tensor(df_pred[self.ts_cols].values, dtype=torch.float32, device=device)
        probs = torch.softmax(self(X), dim=1).cpu().numpy()
        df_pred[[f"prob_{lvl}" for lvl in self.y_levels]] = probs
        return df_pred
    
    def rats_score(self, df_aug: pd.DataFrame, df_src: pd.DataFrame, eps=1e-12) -> pd.DataFrame:
        df_aug = df_aug.copy()          # keep originals intact
        df_src = df_src.reset_index(drop=True)
        df_aug = df_aug.reset_index(drop=True)

        # build a Series of probabilities p(a_tgt | x_hat) and p(a_tgt | x_src)
        p_aug = np.empty(len(df_aug))
        p_src = np.empty(len(df_src))

        # iterate over target classes once, but apply all rows at once (vectorised)
        for lvl in self.y_levels:
            mask = df_aug[self.y_col] == lvl
            col  = f"prob_{lvl}"
            p_aug[mask] = df_aug.loc[mask, col]
            p_src[mask] = df_src.loc[mask, col]

        # RaTS = log( p_aug / p_src )
        df_aug['p_aug'] = p_aug
        df_aug['p_src'] = p_src
        df_aug["RaTS"] = np.log((p_aug + eps) / (p_src + eps))

        return df_aug


def prep_df2pred(
    df2train: pd.DataFrame,  # or df_left
    model, # the vital model object
    config_dict: dict, # the config of vital model
    w: float, # w to interpolate
    y_col: str, # classifier target column (target attribute, i.e. upward trend vs. downward trend)
    *,
    aug_type: str,
    meta: Optional[dict],
    configs: Optional[dict]
):
    # ------------------------------------------------------------------
    # Auto‑detect time-series editing model type (logic unchanged)
    # ------------------------------------------------------------------
    if meta is None:
        model_type = "vital"
    else:
        if "level_maps" in meta:
            model_type = "tedit"
        elif "attr_emb_dim" in meta:
            model_type = "tedit_tx"
        else:
            raise ValueError("Cannot determine model type from meta dictionary")

    model.eval()
    y_levels = list(df2train[y_col].unique()) # levels of the target attribute
    ts_str_cols = [str(i + 1) for i in range(config_dict["seq_length"])] # time-series columns

    if config_dict['open_vocab']:
        aug_level_map = df2train.groupby(y_col).agg({f"{y_col}_aug": lambda x: list(x.unique())}).to_dict()[f"{y_col}_aug"]
    
    df2pred_aug = pd.DataFrame()
    df2pred_src = pd.DataFrame()
    for tgt_level in y_levels:
        df2aug = df2train[df2train[y_col] != tgt_level].copy().reset_index(drop=True)
        if not config_dict['open_vocab']:
            if aug_type == "marginal":
                df2aug["new_text"] = tgt_level
            elif aug_type == "conditional":
                org_levels = list(set(y_levels) - {tgt_level})
                df2aug["new_text"] = df2aug["text"].copy()
                for org_level in org_levels:
                    df2aug["new_text"] = df2aug["new_text"].str.replace(org_level, tgt_level)
        else:
            if aug_type == "marginal":
                df2aug["new_text"] = np.random.choice(aug_level_map[tgt_level], size=len(df2aug))
            elif aug_type == "conditional":
                df2aug[y_col+'_aug'] = np.random.choice(aug_level_map[tgt_level], size=len(df2aug))
                df2aug["new_text"] = ''
                for str_col in [col+'_aug' for col in config_dict["txt2ts_y_cols"]]:
                    df2aug["new_text"] += ' ' + df2aug[str_col]
                df2aug["new_text"] = df2aug["new_text"].str.strip()

        df2aug_src = df2aug[
            ts_str_cols + config_dict["txt2ts_y_cols"] + ["text", "label"]
        ].copy()
        df2pred_src = pd.concat([df2pred_src, df2aug_src], ignore_index=True)

        # -------- generate edited time‑series -------------------------------------------
        col_level_map = {"new_text": tgt_level}
        if meta is not None:
            if model_type == "tedit_tx":
                ts_hat_ls = tedit_tx_generate_ts_tx(
                    df2aug, meta, config_dict, configs, y_col, col_level_map
                )
            else:
                ts_hat_ls = tedit_generate_ts_tx(
                    df2aug, meta, config_dict, configs, y_col, col_level_map
                )
        else:
            ts_hat_ls = interpolate_ts_tx(df2aug, model, config_dict, ["new_text"], w)

        tmp = pd.DataFrame(ts_hat_ls["new_text"], columns=["aug_text", "ts_hat"])
        tmp["ts_hat"] = tmp["ts_hat"].apply(lambda x: x.cpu().detach().numpy())
        df2aug[y_col] = tgt_level
        df2aug[ts_str_cols] = np.vstack(tmp["ts_hat"].to_numpy())

        df2aug = df2aug[ts_str_cols + [y_col] + list(set(config_dict["txt2ts_y_cols"])-set([y_col])) + ["text", "new_text"]].copy()
        df2pred_aug = pd.concat([df2pred_aug, df2aug], ignore_index=True)

    if config_dict.get("ts_global_normalize", False):
        global_mean = config_dict["ts_normalize_mean"]
        global_std = config_dict["ts_normalize_std"]
        df2pred_src[ts_str_cols] = (df2pred_src[ts_str_cols] - global_mean) / global_std

    return df2pred_aug, df2pred_src


def eval_ts_classifier(
    df: pd.DataFrame,  # should be df_left
    model,
    config_dict: dict,
    w: float,
    y_col: str,
    *,
    conditions: Optional[List[Tuple[str, str]]] = None,
    b: Optional[int] = None,
    aug_type: str = "conditional",
    meta: Optional[dict] = None,
    configs: Optional[dict] = None,
):
    
    model.eval()

    clf = TSClassifier(model, df, config_dict, y_col)
    df2train = clf._prepare_df2train(df, config_dict, conditions=conditions, b=b)
    clf.fit(df2train, device=device)


    df2pred_aug, df2pred_src = prep_df2pred(
        df2train,
        model,
        config_dict = config_dict,
        w = w,
        y_col = y_col,
        aug_type=aug_type,
        meta=meta,
        configs=configs,
    )

    df2pred_aug = clf.predict(
        df2pred_aug, device=device
    )
    df2pred_src = clf.predict(
        df2pred_src, device=device
    )
    df2pred_aug = clf.rats_score(df2pred_aug, df2pred_src)
    
    # GPU tidy‑up
    torch.cuda.empty_cache()
    del clf

    # Final packaging
    df2pred_aug["aug_type"] = aug_type
    df2pred_aug["attr"] = y_col
    df2pred_aug["score"] = df2pred_aug["RaTS"]
    df2pred_aug["metric"] = "RaTS"
    df2pred_aug["src_level"] = df2pred_aug["text"]
    df2pred_aug["tgt_level"] = df2pred_aug["new_text"]

    res_df = df2pred_aug[["aug_type", "attr", "src_level", "tgt_level", "metric", "score"]]
    return res_df, None


def plot_rats(df_rats_all, metrics = ['RaTS'], figsize=(15, 4)):
    """
    Create a grid of boxplots comparing RaTS scores across different conditions.
    
    Parameters:
    -----------
    df_rats_all : pd.DataFrame
        DataFrame containing RaTS scores and metadata
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (20, 4)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Get unique conditions including "All"
    conditions = ['All'] + list(df_rats_all.attr.unique())
    n_conditions = len(conditions)

    # Calculate number of rows needed (4 columns per row)
    n_cols = 4
    n_rows = (n_conditions + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with calculated dimensions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    fig.suptitle('RaTS Score Comparison Across Different Conditions', fontsize=14, y=1.02)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot settings
    types_ = ['marginal', "conditional"]

    # First pass: collect all data to determine y-limits
    all_data = []
    for condition in conditions:
        if condition == 'All':
            data = [
                df_rats_all[(df_rats_all.metric == m) & (df_rats_all.aug_type == t)].score
                for m in metrics for t in types_
            ]
        else:
            df_rats = df_rats_all[df_rats_all.attr == condition]
            data = [
                df_rats[(df_rats.metric == m) & (df_rats.aug_type == t)].score
                for m in metrics for t in types_
            ]
        all_data.extend([item for sublist in data for item in sublist])
    
    # Calculate y-limits with some padding
    y_min = min(all_data)
    y_max = max(all_data)
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    y_limits = (y_min - y_padding, y_max + y_padding)

    # Plot each condition
    for idx, condition in enumerate(conditions):
        if condition == 'All':
            # Use all data for "All" condition
            data = [
                df_rats_all[(df_rats_all.metric == m) & (df_rats_all.aug_type == t)].score
                for m in metrics for t in types_
            ]
            title = 'All Conditions'
        else:
            # Filter data for specific condition
            df_rats = df_rats_all[df_rats_all.attr == condition]
            data = [
                df_rats[(df_rats.metric == m) & (df_rats.aug_type == t)].score
                for m in metrics for t in types_
            ]
            title = condition
        
        # Create boxplot with specified configuration
        bp = axes[idx].boxplot(data, 
                            labels=['Marginal', 'Conditional'],
                            notch=True, 
                            showmeans=True,
                            patch_artist=False,  # solid boxes
                            boxprops=dict(color="black"),    # outline of the box
                            whiskerprops=dict(color="black"),
                            capprops=dict(color="black"),
                            medianprops=dict(color="red", linewidth=2),
                            flierprops=dict(markeredgecolor="black", 
                                          markerfacecolor="black", 
                                          marker="o", 
                                          markersize=3))
        
        # Customize axes
        axes[idx].set_title(title, fontsize=12, pad=10)
        axes[idx].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        
        # Set shared y-limits
        axes[idx].set_ylim(y_limits)
        
        # Only show ylabel for the first subplot in each row
        if idx % n_cols == 0:
            axes[idx].set_ylabel('RaTS Score', fontsize=10)
        
        axes[idx].tick_params(axis='x', rotation=0)

    # Hide any unused subplots
    for idx in range(len(conditions), len(axes)):
        axes[idx].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    return fig


def eval_rats(df,
              model,
              config_dict,
              y_edit,
              w,
              *,
              aug_type: str = "conditional",
              meta: Optional[dict] = None,
              configs: Optional[dict] = None,
              ):
    # ---------------- load/train all classifiers ----------------
    clfs = {}
    for y_col in config_dict['txt2ts_y_cols']:
        clf_path = config_dict['output_dir'] + f'/rats_clf_{y_col}.pth'
        clf = TSClassifier(model, df, config_dict, y_col).to(device)
        if not os.path.exists(clf_path):
            df2train = clf._prepare_df2train(df, config_dict, conditions=None, b=None)
            clf.fit(df2train, device=device)
            torch.save(clf.state_dict(), clf_path)
        else:
            clf.load_state_dict(torch.load(clf_path, map_location=device))
        clfs[y_col] = clf
    
    # ---------------- edit time series towards y_edit ----------------
    df2train = clfs[y_edit]._prepare_df2train(df, config_dict, conditions=None, b=None)
    df2pred_aug, df2pred_src = prep_df2pred(
        df2train,
        model,
        config_dict = config_dict,
        w = w,
        y_col = y_edit,
        aug_type=aug_type,
        meta=meta,
        configs=configs,
    )

    # ---------------- calculate RaTS ----------------
    # edited
    clf = clfs[y_edit]
    df_pred_aug = clf.predict(df2pred_aug, device=device)
    df_pred_src = clf.predict(df2pred_src, device=device)
    df_rats = clf.rats_score(df_pred_aug, df_pred_src)[['p_aug', 'p_src', 'RaTS', 'text', 'new_text']]
    # preserved
    y_preserve = list(set(config_dict['txt2ts_y_cols'])-set([y_edit]))
    for y_pres in y_preserve:
        clf = clfs[y_pres]
        df_pred_aug = clf.predict(df2pred_aug, device=device)
        df_pred_src = clf.predict(df2pred_src, device=device)
        df_rats_pres = clf.rats_score(df_pred_aug, df_pred_src)[['p_aug', 'p_src', 'RaTS']]
        df_rats_pres.columns = [f"{col}_{y_pres}" for col in df_rats_pres.columns]
        df_rats = pd.concat([df_rats, df_rats_pres], axis=1) # concate by columns

    torch.cuda.empty_cache()
    del clfs, clf
    df_rats_eng = eng_df_rats(df_rats, aug_type, y_edit, y_preserve)
    return df_rats, df_rats_eng


def eng_df_rats(df_rats, aug_type, y_edit, y_preserve):
    # score columns for the edited attribute
    df_rats["aug_type"] = aug_type
    df_rats["attr"] = y_edit
    df_rats["score"] = df_rats["RaTS"]
    df_rats["metric"] = "RaTS"
    df_rats["src_level"] = df_rats["text"]
    df_rats["tgt_level"] = df_rats["new_text"]
    df_rats_edit = df_rats[["aug_type", "attr", "src_level", "tgt_level", "metric", "score"]]

    # average across score columns for the preserved attributes
    df_rats['RaTS_preserved'] = df_rats[[f"RaTS_{y}" for y in y_preserve]].abs().mean(axis=1)
    df_rats_pres = df_rats[["aug_type", "attr", "src_level", "tgt_level", "RaTS_preserved"]]
    df_rats_pres['score'] = df_rats_pres['RaTS_preserved']
    df_rats_pres['metric'] = "RaTS_preserved"
    df_rats_pres = df_rats_pres[["aug_type", "attr", "src_level", "tgt_level", "metric", "score"]]

    # final packaging
    df_rats_eng = pd.concat([df_rats_edit, df_rats_pres], ignore_index=True)
    return df_rats_eng
    
    