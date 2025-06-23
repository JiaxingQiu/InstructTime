import torch
import torch.nn.functional as F
import random
import numpy as np
import time



def compute_clip_loss(logits, labels, targets, 
                      target_type = 'by_target', 
                      ts_embedded=None, text_embedded=None): # embedded or features
    if target_type == 'by_label': # diagonal target matrix
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        targets = labels_equal.float()
        loss_ts = cross_entropy(logits, targets)
        loss_tx = cross_entropy(logits.T, targets.T)
        clip_loss = (loss_ts + loss_tx) / 2

    if target_type == 'by_target': # use external target matrix
        loss_ts = cross_entropy(logits, targets)
        loss_tx = cross_entropy(logits.T, targets.T)
        clip_loss = (loss_ts + loss_tx) / 2
    
    if target_type == 'by_similarity': # use current step calculated similarity matrix
        batch_size = logits.shape[0]
        with torch.no_grad(): 
            targets = get_similarity_target(ts_embedded, text_embedded)
        assert targets.shape == (batch_size, batch_size)
        loss_ts = cross_entropy(logits, targets)
        loss_tx = cross_entropy(logits.T, targets.T)
        clip_loss = (loss_ts + loss_tx) / 2
        
    return clip_loss

def cross_entropy(preds, targets):
    # Compute cross entropy loss manually
    batch_size = preds.shape[0]
    exp_preds = torch.exp(preds)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # sum over the columns, keep the rows
    log_probs = torch.log(softmax_probs + 1e-16)
    loss = -torch.sum(targets * log_probs) / batch_size
    return loss

def get_similarity_target(ts_embedded, text_embedded):
    ts_embedded = F.normalize(ts_embedded, p=2, dim=1)  # dim=1 for row-wise normalization
    text_embedded = F.normalize(text_embedded, p=2, dim=1)
    ts_similarity = torch.matmul(ts_embedded, ts_embedded.T) # cosine similarity
    texts_similarity = torch.matmul(text_embedded, text_embedded.T) # cosine similarity
    target = (ts_similarity + texts_similarity)/2
    return target

# class KLAnnealer:
#     def __init__(self, start, end, epochs):
#         self.start = start
#         self.end = end#min(end,1)
#         self.epochs = epochs
        
#     def get_beta(self, epoch):
#         beta = min(self.end, self.start + (self.end - self.start) * epoch / self.epochs)
#         # beta = np.exp(beta)-1
#         return beta

# def compute_vae_loss(ts, ts_hat, mean, log_var, beta=1.0):
#     reconstruction_loss = F.mse_loss(ts_hat, ts, reduction='sum') / ts.size(0)
#     kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
#     vae_loss = reconstruction_loss + beta * kl_loss
#     # print(f'reconstruction: {reconstruction_loss.item()}, kl: {kl_loss.item()}')
#     return vae_loss


def compute_reconstruction_loss(ts, ts_hat):
    reconstruction_loss = F.mse_loss(ts_hat, ts, reduction='sum') / ts.size(0)
    return reconstruction_loss

def compute_kl_loss(mean, log_var):
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
    return kl_loss


def compute_loss(model, ts, text_features, labels, targets, target_type = 'by_target', train_type='joint', alpha=1.0, beta=0.1):
    # initialized to return
    loss = 0
    clip_loss = 0
    reconstruction_loss = 0
    kl_loss = 0

    logits, ts_hat, mean, log_var = model(ts, text_features)
    # compute losses
    if train_type == 'joint':
        clip_loss = compute_clip_loss(logits, labels, targets, target_type)
        reconstruction_loss =  alpha * compute_reconstruction_loss(ts, ts_hat)
        kl_loss = beta * compute_kl_loss(mean, log_var)
        loss = clip_loss + reconstruction_loss + kl_loss
    
    elif train_type == 'vae':
        reconstruction_loss = alpha * compute_reconstruction_loss(ts, ts_hat)
        kl_loss = beta * compute_kl_loss(mean, log_var)  
        loss = reconstruction_loss + kl_loss
    
    elif train_type == 'clip':
        clip_loss = compute_clip_loss(logits, labels, targets, target_type)
        kl_loss = beta * compute_kl_loss(mean, log_var)
        loss = clip_loss + kl_loss
    
    else:
        raise ValueError(f"Invalid model type: {train_type}")
    
    return loss, clip_loss, reconstruction_loss, kl_loss # alpha and beta are already applied



def train_epoch(model, train_dataloader, optimizer, target_type = 'by_target', train_type='joint', alpha=1.0, beta=1.0): 
    # alpha controls the balance between vae loss and clip loss
    # beta controls the balance between reconstruction loss and kl loss
    model.train()
    total_loss = 0
    total_clip_loss = 0 
    total_reconstruction_loss = 0
    total_kl_loss = 0 
    num_batches = 0
    
    for _, (idx, ts, text_features, labels, targets) in enumerate(train_dataloader):
        targets = targets[:,idx]
        loss, clip_loss, reconstruction_loss, kl_loss = compute_loss(model, ts, text_features, labels, targets, target_type, train_type, alpha, beta)

        # Clear gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward(retain_graph=True)
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update weights
        optimizer.step()

        # update total loss and number of batches
        total_loss += loss.item()
        if train_type == 'joint':
            total_clip_loss += clip_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_clip_loss = total_clip_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_clip_loss, avg_reconstruction_loss, avg_kl_loss



def test_epoch(model, test_dataloader, target_type = 'by_target', train_type='joint', alpha=1.0, beta=0.1):
    model.eval()
    total_loss = 0
    total_clip_loss = 0 
    total_reconstruction_loss = 0 
    total_kl_loss = 0 
    num_batches = 0
    
    with torch.no_grad():
        for _, (idx, ts, text_features, labels, targets) in enumerate(test_dataloader):
            targets = targets[:,idx]
            loss, clip_loss, reconstruction_loss, kl_loss = compute_loss(model, ts, text_features, labels, targets, target_type, train_type, alpha, beta)
            
            total_loss += loss.item()
            if train_type == 'joint':
                total_clip_loss += clip_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_kl_loss += kl_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_clip_loss = total_clip_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_clip_loss, avg_reconstruction_loss, avg_kl_loss


def train_vital(model, train_dataloader, test_dataloader, optimizer, scheduler, num_epochs, 
                target_type = 'by_target', train_type='joint', alpha_init=None, beta = 1.0,
                es_patience=500, target_ratio=100):
    
    # Set random seeds for reproducibility
    torch.manual_seed(333)
    random.seed(333)
    np.random.seed(333)

    train_losses = []
    test_losses = []

    # Keep track of best model and early stopping
    best_test_loss = float('inf')
    best_model_state = None
    counter = 0
    alpha = alpha_init if alpha_init is not None else 1.0
    
    try:
        for epoch in range(num_epochs):
            start_time = time.time()
            # Train and test for one epoch
            train_loss, train_clip_loss, train_reconstruction_loss, train_kl_loss = train_epoch(model, train_dataloader, optimizer, target_type, train_type, alpha, beta)
            test_loss, test_clip_loss, test_reconstruction_loss, test_kl_loss = test_epoch(model, test_dataloader, target_type, train_type, alpha, beta)
            
            # Recalibrate alpha
            if (epoch == 10 and alpha_init is None and train_type == 'joint'):
                alpha = recalibrate_alpha(train_clip_loss, train_reconstruction_loss/alpha, target_ratio=target_ratio)
                print("-"*100)
                print(f"\nRecalibrating alpha to: {alpha:.2e}")
                print("-"*100)
            
            # Store losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Update learning rate
            scheduler.step(test_loss)
            
            # Save best model and check early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= es_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs. No improvement in test loss for {es_patience} epochs.")
                    break
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Early stopping if learning rate becomes too small
            if current_lr <= scheduler.min_lrs[0]:
                print("Learning rate too small. Stopping training.")
                break

            # Print progress
            if epoch % 10 == 0:
                epoch_time = time.time() - start_time
                if train_type == 'joint':
                    print(f'Epoch [{epoch}/{num_epochs}] {epoch_time:.2f}s')
                    print(f'\tTraining Loss: {train_loss:.6f} (clip: {train_clip_loss:.6f}, rc: {train_reconstruction_loss:.6f}, kl: {train_kl_loss:.6f})')
                    print(f'\tTesting Loss: {test_loss:.6f} (clip: {test_clip_loss:.6f}, rc: {test_reconstruction_loss:.6f}, kl: {test_kl_loss:.6f})')
                    print(f'\tLearning Rate: {current_lr:.9f}')
                    print(f'\talpha: {alpha}, beta: {beta}')
                else:
                    print(f'Epoch [{epoch}/{num_epochs}] {epoch_time:.2f}s')
                    print(f'\tTraining Loss: {train_loss:.6f}')
                    print(f'\tTesting Loss: {test_loss:.6f}')
                    print(f'\tLearning Rate: {current_lr:.9f}')
                    print(f'\talpha: {alpha}, beta: {beta}')
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    except Exception as e:
        raise e
    finally:
        # Load best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return train_losses, test_losses, alpha



def recalibrate_alpha(clip_loss, reconstruction_loss, target_ratio=10):
    """
    Calculate alpha to make reconstruction_loss * alpha approximately target_ratio times smaller than clip_loss
    
    Example:
    If clip_loss = 900 (9 * 10^2)
    and reconstruction_loss = 2000 (2 * 10^3)
    and you want reconstruction_loss to be 1000x smaller than clip_loss:
    
    clip_loss = 9 * 10^2
    reconstruction_loss = 2 * 10^3
    target_ratio = 1000
    
    Then alpha should be approximately 10^-4 to make:
    reconstruction_loss * alpha ≈ 2 * 10^3 * 10^-4 = 2 * 10^-1
    which is 1000x smaller than clip_loss (9 * 10^2)
    """
    if target_ratio is None:
        return 1.0
    # Get the order of magnitude (power of 10) for each loss
    clip_power = int(np.log10(clip_loss))
    recon_power = int(np.log10(reconstruction_loss))
    
    # Calculate the difference in powers
    power_diff = clip_power - recon_power
    
    # Calculate alpha to achieve target ratio
    alpha = 10 ** -(np.log10(target_ratio) - power_diff)
    alpha = min(alpha, 1.0)
    return alpha
