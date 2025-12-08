import os, time, torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Same training / validation loop as PreTrainer_wo_c,
    but the external evaluator is removed—so it *only*
    trains and tracks validation loss.
    """
    # ---------- init ----------
    def __init__(self, configs, model, train_loader, valid_loader):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_opt()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # best-loss tracking & logging
        self._best_valid_loss = 1e10
        self._global_batch_no = 0
        self.tf_writer = SummaryWriter(log_dir=self.output_folder)

    # ---------- helpers ----------
    def _init_cfgs(self, configs):
        self.configs = configs
        self.n_epochs             = configs["epochs"]
        self.itr_per_epoch        = configs["itr_per_epoch"]
        self.valid_epoch_interval = configs["val_epoch_interval"]
        self.display_epoch_interval = configs["display_interval"]

        self.lr         = configs["lr"]
        self.batch_size = configs["batch_size"]
        self.model_path = configs["model_path"]
        self.output_folder = configs["output_folder"]
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model
        if self.model_path:
            print(f"Loading pretrained model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_opt(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

    # ---------- public API ----------
    def train(self):
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            if (epoch + 1) % self.valid_epoch_interval == 0:
                self._validate(epoch)

    # ---------- internal ----------
    def _train_epoch(self, epoch_no):
        start_time = time.time()
        self.model.train()
        running = 0.0

        for batch_no, batch in enumerate(self.train_loader):
            self._global_batch_no += 1

            self.opt.zero_grad()
            loss = self.model(batch, is_train=True, mode="pretrain")
            loss.backward()
            self.opt.step()

            running += loss.item()
            self.tf_writer.add_scalar("Train/batch_loss", loss.item(), self._global_batch_no)

            if batch_no >= self.itr_per_epoch:
                break

        epoch_loss = running / len(self.train_loader)
        self.tf_writer.add_scalar("Train/epoch_loss", epoch_loss, epoch_no)

        if (epoch_no + 1) % self.display_epoch_interval == 0:
            print(f"Epoch {epoch_no} | Loss {epoch_loss:.4f} | "
                  f"Time {time.time() - start_time:.2f}s")

    @torch.no_grad()
    def _validate(self, epoch_no):
        self.model.eval()
        total = 0.0
        for batch in self.valid_loader:
            total += self.model(batch, is_train=False).item()

        avg_loss = total / len(self.valid_loader)
        self.tf_writer.add_scalar("Valid/epoch_loss", avg_loss, epoch_no)

        if avg_loss < self._best_valid_loss:
            self._best_valid_loss = avg_loss
            print(f"\n*** New best val-loss {avg_loss:.4f} at epoch {epoch_no}\n")
            self._save_model(epoch_no)

    def _save_model(self, epoch_no):
        ckpt_dir = os.path.join(self.output_folder, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))
        torch.save(self.model.state_dict(),
                   os.path.join(ckpt_dir, f"model_best_{epoch_no}.pth"))


        
# train_configs = {
#     # Training hyperparameters
#     "epochs": 100,
#     "itr_per_epoch": 100,            # max number of batches per epoch
#     "val_epoch_interval": 5,         # how often to validate
#     "display_interval": 1,           # how often to print training logs
#     "lr": 1e-4,
#     "batch_size": 64,

#     # Paths
#     "model_path": None,              # or provide a path like "checkpoints/model_x.pth"
#     "output_folder": "runs/pretrain" # where logs and checkpoints go
# }
