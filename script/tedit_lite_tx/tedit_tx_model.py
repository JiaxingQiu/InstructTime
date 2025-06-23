import torch
import torch.nn as nn

from models_tx.diffusion.diff_csdi import Diff_CSDI
from models_tx.diffusion.diff_csdi_multipatch import Diff_CSDI_Patch, Diff_CSDI_MultiPatch, Diff_CSDI_MultiPatch_Parallel
from models_tx.diffusion.diff_csdi_time_weaver import Diff_CSDI_TimeWeaver
from models_tx.diffusion.diff_csdi_multipatch_weaver import Diff_CSDI_MultiPatch_Weaver_Parallel
from models_tx.encoders.attr_encoder import AttributeEncoder
from models_tx.encoders.side_encoder import SideEncoder
from models_tx.samplers import DDPMSampler, DDIMSampler


class ConditionalGenerator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]

        # # Add text embedding dimension to configs
        # if "text_emb_dim" not in configs["attrs"]:
        #     configs["attrs"]["text_emb_dim"] = 768  # Default BERT embedding size
            
        self._init_condition_encoders(configs["attrs"], configs["side"])
        self._init_diff(configs["diffusion"])

    def _init_condition_encoders(self, configs_attr, configs_side):
        configs_attr["device"] = self.device
        configs_side["device"] = self.device
        self.attr_en = AttributeEncoder(configs_attr).to(self.device)
        self.side_en = SideEncoder(configs_side).to(self.device)

    def _init_diff(self, configs):
        configs["side_dim"] = self.side_en.total_emb_dim
        configs["attr_dim"] = self.attr_en.emb_dim
        configs["n_attrs"] = self.attr_en.n_attrs

        # input_dim = 1 if self.is_unconditional == True else 2
        input_dim = 1
        if configs["type"] == "CSDI":
            self.diff_model = Diff_CSDI(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_Patch":
            self.diff_model = Diff_CSDI_Patch(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch":
            self.diff_model = Diff_CSDI_MultiPatch(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch_Parallel":
            self.diff_model = Diff_CSDI_MultiPatch_Parallel(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_MultiPatch_Weaver_Parallel":
            self.diff_model = Diff_CSDI_MultiPatch_Weaver_Parallel(configs, input_dim).to(self.device)
        if configs["type"] == "CSDI_TimeWeaver":
            self.diff_model = Diff_CSDI_TimeWeaver(configs, input_dim).to(self.device)

        # steps
        self.num_steps = configs["num_steps"]

        # sampler
        self.ddpm = DDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)

        # edit
        self.edit_steps = configs["edit_steps"]
        self.bootstrap_ratio = configs["bootstrap_ratio"]

    def predict_noise(self, xt, side_emb, src_attr_emb, t):
        noisy_x = torch.unsqueeze(xt, 1)  # (B,1,K,L): this is required by the diff model
        pred_noise = self.diff_model(noisy_x, side_emb, src_attr_emb, t)  # (B,K,L)
        return pred_noise

    def _noise_estimation_loss(self, x, side_emb, attr_emb, t):        
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        pred_noise = self.predict_noise(noisy_x, side_emb, attr_emb, t)
        
        residual = noise - pred_noise
        loss = (residual ** 2).mean()
        return loss
    
    def forward(self, batch, is_train=False, mode="pretrain"):
        """
        Training.
        """
        if mode == "pretrain":
            return self.pretrain(batch, is_train)
        elif mode == "finetune":
            return self.fintune(batch, is_train)
    
    """
    Pretrain.
    """
    def pretrain(self, batch, is_train):
        x, tp, attrs = self._unpack_data_cond_gen(batch)
    
        side_emb = self.side_en(tp)
        attr_emb = self.attr_en(attrs)
        B = x.shape[0]

        if is_train:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
            return self._noise_estimation_loss(x, side_emb, attr_emb, t)
        
        # valid
        loss = 0
        for t in range(self.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            loss += self._noise_estimation_loss(x, side_emb, attr_emb, t)
        return loss/self.num_steps

    def _unpack_data_cond_gen(self, batch):
        x = batch["x"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        attrs = batch["attrs"].to(self.device).long()
    
        x = x.permute(0, 2, 1)  # (B,L,K) -> (B,K,L)
        return x, tp, attrs

    """
    Finetune.
    """

    def fintune(self, batch, is_train):
        src_x, tp, src_attrs, tgt_attrs, tgt_x = self._unpack_data_edit(batch)

        side_emb = self.side_en(tp)
        src_attr_emb = self.attr_en(src_attrs)
        tgt_attr_emb = self.attr_en(tgt_attrs)

        # src -> tgt
        tgt_x_pred = self._edit(src_x, side_emb, src_attr_emb, tgt_attr_emb, sampler="ddim")  # (B,V,L)
        if self.bootstrap_ratio > 0:
            bs_ids = self._bootstrap(tgt_x_pred, src_x, side_emb, src_attr_emb, tgt_attr_emb)  # (B)
        else:
            bs_ids = None

        if is_train:
            # tgt
            if bs_ids is not None:
                t = torch.randint(0, self.num_steps, [len(bs_ids)], device=self.device)
                loss_tgt = self._noise_estimation_loss(tgt_x_pred[bs_ids], side_emb[bs_ids], tgt_attr_emb[bs_ids], t)
            else:
                loss_tgt = 0.0
        else:
            loss_tgt = 0
            B = tgt_x_pred.shape[0]
            for t in range(self.num_steps):
                t = (torch.ones(B, device=self.device) * t).long()
                if self.bootstrap_ratio > 0:
                    loss_tgt += self._noise_estimation_loss(tgt_x_pred, side_emb, tgt_attr_emb, t)
            loss_tgt = loss_tgt/self.num_steps
        return loss_tgt

    def _bootstrap(self, tgt_x_pred, src_x, side_emb, src_attr_emb, tgt_attr_emb):
        """
        Translate tgt_x_pred back to src_x_pred.
        Calculate similarity score between src_x and src_x_pred as the confidence score for tgt_x_pred.
        Return the idx of top bootstrap_ratio samples.
        """
        B = src_x.shape[0]
        tgt_x_pred.detach()

        with torch.no_grad():
            src_x_pred = self._edit(tgt_x_pred, side_emb, tgt_attr_emb, src_attr_emb, sampler="ddim")
            src_pred = src_x_pred.detach()
        score = -torch.sum(torch.sum((src_pred - src_x)**2, dim=-1), dim=-1)  # (B)
        
        B_bs = int(B*self.bootstrap_ratio)
        ids = torch.topk(score, B_bs, dim=0)[1]  # select top B_bs samples
        return ids        

    def _unpack_data_edit(self, batch):
        """
        Make the edit-mode batch robust:
          • If `src_*` keys are missing, fall back to `x` / `attrs`.
          • If `tgt_x` is missing, create a zero tensor of the same shape as `x`.
          • If `tgt_attrs` is missing, reuse `attrs`.
        Returns
        -------
        src_x, tp, src_attrs, tgt_attrs, tgt_x  (all on self.device)
        """
        # fallback substitutions ------------------------------------------------
        if "src_x" not in batch:
            batch["src_x"] = batch["x"]
        if "src_attrs" not in batch:
            batch["src_attrs"] = batch["attrs"]

        if "tgt_x" not in batch:
            batch["tgt_x"] = torch.zeros_like(batch["x"])
        if "tgt_attrs" not in batch:
            batch["tgt_attrs"] = batch["attrs"]

        # move & reshape --------------------------------------------------------
        src_x     = batch["src_x"].to(self.device).float().permute(0, 2, 1)  # (B,L,K) → (B,K,L)
        tgt_x     = batch["tgt_x"].to(self.device).float().permute(0, 2, 1)
        src_attrs = batch["src_attrs"].to(self.device).long()
        tgt_attrs = batch["tgt_attrs"].to(self.device).long()
        tp        = batch["tp"].to(self.device).float() if "tp" in batch else None

        return src_x, tp, src_attrs, tgt_attrs, tgt_x

    """
    Generation.
    """
    @torch.no_grad()
    def generate(self, batch, n_samples, mode="edit", sampler="ddpm"): # 
        return self.__getattribute__(mode)(batch, n_samples, sampler)

    def cond_gen(self, batch, n_samples, sampler="ddpm"):
        src_x, tp, src_attrs, tgt_attrs, tgt_x = self._unpack_data_edit(batch)

        side_emb = self.side_en(tp)
        attr_emb = self.attr_en(tgt_attrs)

        samples = []
        B = src_x.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(src_x)
            # x = src_x
            for t in range(self.num_steps-1, -1, -1):
                noise = torch.randn_like(x)  # noise for std
                pred_noise = self.predict_noise(x, side_emb, attr_emb, t)
                t = (torch.ones(B, device=self.device) * t).long()
                if sampler == "ddpm":
                    x = self.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)
    
    def edit(self, batch, n_samples, sampler="ddpm-ddim"):
        """
        Args:
           sampler: forward-backward: ddim-ddim or ddim. 
        """
        src_x, tp, src_attrs, tgt_attrs, tgt_x = self._unpack_data_edit(batch)
        src_attrs = torch.zeros_like(src_attrs) # turn off source attrs

        side_emb = self.side_en(tp)
        src_attr_emb = self.attr_en(src_attrs)
        tgt_attr_emb = self.attr_en(tgt_attrs)

        samples = []        
        for i in range(n_samples):
            tgt_x_pred = self._edit(src_x, side_emb, src_attr_emb, tgt_attr_emb, sampler)
            samples.append(tgt_x_pred)
        return torch.stack(samples)
    
    def _edit(self, src_x, side_emb, src_attr_emb, tgt_attr_emb, sampler):
        B = src_x.shape[0]

        # forward
        xt = src_x
        if sampler[:4] == "ddpm":
            noise = torch.randn_like(src_x)
            xt = self.ddpm.forward(xt, self.edit_steps-1, noise=noise)
        else:
            for t in range(-1, self.edit_steps-1):
                if t == -1:
                    pred_noise = 0
                    t = (torch.ones(B, device=self.device) * t).long()
                else:
                    t = (torch.ones(B, device=self.device) * t).long()
                    pred_noise = self.predict_noise(xt, side_emb, src_attr_emb, t)
                xt = self.ddim.forward(xt, pred_noise, t)

        # reverse
        for t in range(self.edit_steps-1, -1, -1):
            noise = torch.randn_like(xt)
            t = (torch.ones(B, device=self.device) * t).long()
            pred_noise = self.predict_noise(xt, side_emb, tgt_attr_emb, t)
            if sampler[-4:] == "ddpm":
                xt = self.ddpm.reverse(xt, pred_noise, t, noise)
            else:
                xt = self.ddim.reverse(xt, pred_noise, t, noise, is_determin=True)
        return xt
    
    
    
    
# model_configs = {
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
    
#     # Attribute encoder config
#     "attrs": {
#         "attr_dims": [10, 5],         # e.g., two attributes with vocab sizes 10 and 5
#         "emb_dim": 16,
#         "hidden_dims": [64, 64],
#         "dropout": 0.1
#     },

#     # Side encoder config
#     "side": {
#         "input_dim": 1,
#         "seq_len": 50,
#         "emb_dim": 16,
#         "hidden_dims": [64, 64],
#         "dropout": 0.1
#     },

#     # Diffusion model config
#     "diffusion": {
#         "type": "CSDI_MultiPatch_Weaver_Parallel",  # or any other supported type
#         "schedule": "linear",
#         "beta_start": 0.0001,
#         "beta_end": 0.02,
#         "num_steps": 50,
#         "edit_steps": 25,
#         "bootstrap_ratio": 0.5,
#         "patch_size": 10,
#         "n_layers": 4,
#         "d_model": 64,
#         "n_heads": 4,
#         "side_dim": None,   # will be filled in during init
#         "attr_dim": None,   # will be filled in during init
#         "n_attrs": None     # will be filled in during init
#     }
# }



