import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import gc
import os
from datetime import datetime
from utils import print_gpu_memory, collate_fn
from logger import log_batch_predictions

def train_model(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    batch_size=4,
    num_epochs=20,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    grad_accum_steps=8,
    max_len=25,
    log_predictions=True,
    log_frequency=10,
    stage_name="",
    checkpoint_interval=100
):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}: {param.numel()}")

    log_file = f"training_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join("logs", log_file)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step,epoch,batch,loss\n")

    print("[Before empty_cache()]")
    print_gpu_memory()
    torch.cuda.empty_cache()
    print("[After empty_cache()]")
    print_gpu_memory()
    gc.collect()

    model.train()
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}] Starting")
        print_gpu_memory()
        total_loss = 0.0
        step_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        loss_print_interval = max(len(train_loader) // 10, 1)  # ~10x per epoch
        prediction_log_interval = max(len(train_loader) // 5, 1)  # ~5x per epoch

        for batch_idx, batch in enumerate(pbar):
            try:
                protein_data = batch['protein_embeddings'].to(device)
                drug_graph = batch['drug_graphs'].to(device)
                target_text = batch['encoded_texts']
                if target_text:
                    target_text = {k: v.to(device) for k, v in target_text.items()}

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model(protein_data, drug_graph, target_text)
                    loss = loss / grad_accum_steps

                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    avg_loss = total_loss / step_count if step_count > 0 else 0
                    with open(log_path, "a") as f:
                        f.write(f"{step_count},{epoch+1},{batch_idx+1},{avg_loss:.4f}\n")

                total_loss += loss.item() * grad_accum_steps
                step_count += 1

                # Minimal console loss printing
                if batch_idx % loss_print_interval == 0 and step_count > 0:
                    avg_loss = total_loss / step_count
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {avg_loss:.4f}")

                # Periodically log qualitative predictions
                if log_predictions and batch_idx % prediction_log_interval == 0:
                    model.eval()
                    try:
                        log_batch_predictions(model, batch, tokenizer, device, batch_idx, epoch)
                    except Exception as e:
                        print(f"[Prediction log error] {e}")
                    model.train()

                if batch_idx % (len(train_loader) // 5) == 0:
                    torch.cuda.empty_cache()

                if (batch_idx + 1) % checkpoint_interval == 0:
                    ckpt_path = f"checkpoints/ckpt_epoch{epoch}_step{step_count}_{stage_name}_batch{batch_idx + 1}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step_count
                    }, ckpt_path)
                    print(f"[Checkpoint saved] {ckpt_path}")

            except RuntimeError as e:
                if "out of memory" in str(e) or "device-side assert" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_epoch_loss = total_loss / step_count if step_count > 0 else 0
        print(f"[Epoch {epoch+1}] Finished. Avg loss: {avg_epoch_loss:.4f}")
        torch.cuda.empty_cache()

        final_ckpt = f"checkpoints/ckpt_epoch{epoch}_final_{stage_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': step_count
        }, final_ckpt)
        print(f"[Checkpoint saved] {final_ckpt}")

    return model