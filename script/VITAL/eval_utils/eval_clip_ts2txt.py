# ---------------- functions to evaluate CLIP in terms of multiple text classification -------------------- 
import torch
from .eval_clip import *

class EvalCLIPTS2TXT:
    def __init__(self, 
                 df_new,
                 config_dict,
                 y_true_cols = ['true1', 'true2'], 
                 y_pred_cols = ['text1', 'text2'],
                 y_pred_cols_ls = [['demo', 'text1', 'ts_description'], 
                                   ['demo', 'text2', 'ts_description']]):
        # true1 column is one-hot indicator of true text of first level of outcome, "this infant will die in 7 days"
        # true2 column is one-hot indicator of true text of second level of outcome, "this infant will survive"
        # text1 column is text of first level of predicted outcome, "this infant will die in 7 days"
        # text2 column is text of second level of predicted outcome, "this infant will survive"

        self.y_true = torch.tensor(df_new[y_true_cols].values)
        self.y_true = self.y_true.to(device)
        if not config_dict['3d']:
            self.ts_f_mat, self.tx_f_mat_ls, _ = get_features3d(df_new, 
                                                                config_dict,
                                                                text_col_ls=y_pred_cols) # text1 is a column of "will die", text2 is a column of "will survive"
            self.ts_f_mat = self.ts_f_mat.to(device)
            self.tx_f_mat_ls = [tx_f_mat.to(device) for tx_f_mat in self.tx_f_mat_ls]
        
        elif config_dict['3d']:
            tx_f_mat_ls_ls = []
            for y_pred_cols in y_pred_cols_ls:
                # replace cl_event with text1 / text2 in tx_df_mat_ls
                ts_f_mat, tx_f_mat_ls, _ = get_features3d(df_new,
                                                          config_dict,
                                                          text_col_ls=y_pred_cols) # text1 is a column of "will die", text2 is a column of "will survive"
                tx_f_mat_ls = [tx_f_mat.to(device) for tx_f_mat in tx_f_mat_ls]
                tx_f_mat_ls_ls.append(tx_f_mat_ls)
            self.tx_f_mat_ls_ls = tx_f_mat_ls_ls
            self.ts_f_mat = ts_f_mat
            self.ts_f_mat = self.ts_f_mat.to(device)

        
        
@torch.no_grad() 
def eval_clip_ts2txt(model, 
              eval_inputs,
              return_probs=False,
              batch_size=32):  # Process logits in batches
    
    model.eval()
    with torch.no_grad(): 
        y_true = eval_inputs.y_true
        ts_f_mat = eval_inputs.ts_f_mat
        tx_f_mat_ls = eval_inputs.tx_f_mat_ls
        
        # Process in batches to save memory
        num_samples = ts_f_mat.size(0)
        obs_ys_logits = []
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            ts_f_batch = ts_f_mat[i:batch_end]
            
            batch_logits = []
            for tx_f_mat in tx_f_mat_ls:
                logit, _, _, _ = model(ts_f_batch, tx_f_mat[i:batch_end])
                # Keep the diagonal of logit
                obs_logits = torch.diag(logit)
                obs_logits = obs_logits.reshape(-1, 1)
                batch_logits.append(obs_logits)
                del logit, obs_logits
                torch.cuda.empty_cache()
            
            # Concatenate batch logits
            batch_logits = torch.cat(batch_logits, dim=1)
            obs_ys_logits.append(batch_logits)
            del batch_logits
            torch.cuda.empty_cache()
        
        # Combine all batches
        obs_ys_logits = torch.cat(obs_ys_logits, dim=0)
        
        # Calculate probabilities
        exp_preds = torch.exp(obs_ys_logits)
        softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)
        y_prob = softmax_probs

        eval_metrics = get_eval_metrics(y_true, y_prob)
        
        # Delete large tensors
        del obs_ys_logits, exp_preds, softmax_probs
        torch.cuda.empty_cache()

    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics

@torch.no_grad() 
def eval_clip3d_ts2txt(model, # model of CLIP3DModel
                eval_inputs,
                return_probs=False,
                batch_size=32):  # Process logits in batches
    model.eval()
    with torch.no_grad(): 
        y_true = eval_inputs.y_true
        ts_f_mat = eval_inputs.ts_f_mat
        tx_f_mat_ls_ls = eval_inputs.tx_f_mat_ls_ls

        # Process in batches to save memory
        num_samples = ts_f_mat.size(0)
        obs_ys_logits = []
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            ts_f_batch = ts_f_mat[i:batch_end]
            
            batch_logits = []
            for tx_f_mat_ls in tx_f_mat_ls_ls:
                logits, _, _, _ = model(ts_f_batch, [tx_f_mat[i:batch_end] for tx_f_mat in tx_f_mat_ls])
                obs_logits = torch.diag(logits)
                obs_logits = obs_logits.reshape(-1, 1)
                batch_logits.append(obs_logits)
                del logits, obs_logits
                torch.cuda.empty_cache()
            
            # Concatenate batch logits
            batch_logits = torch.cat(batch_logits, dim=1)
            obs_ys_logits.append(batch_logits)
            del batch_logits
            torch.cuda.empty_cache()
        
        # Combine all batches
        obs_ys_logits = torch.cat(obs_ys_logits, dim=0)
        
        # Calculate probabilities
        exp_preds = torch.exp(obs_ys_logits)
        y_prob = exp_preds / exp_preds.sum(dim=1, keepdim=True)

        eval_metrics = get_eval_metrics(y_true, y_prob)

        del obs_ys_logits, exp_preds
        torch.cuda.empty_cache()

    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics


