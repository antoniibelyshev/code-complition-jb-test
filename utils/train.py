import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def train_model(
    model,
    tokenizer,
    train_dataset,
    batch_size = 2,
    start_epoch = 0,
    end_epoch = 6,
    learning_rate = 1e-4,
):
    train_loader = DataLoader(torch.arange(len(train_dataset)), batch_size=batch_size, shuffle=True)

    # freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers[-5:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    run = wandb.init(entity="antonii-belyshev", project="finetune")
    # Fine-tuning loop
    for epoch in range(start_epoch, end_epoch):
        model.train()
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for batch in progress_bar:
            tokens = tokenizer([train_dataset[i] for i in batch], padding=True, truncation=True)
            del batch

            input_ids = [tokens_seq + [tokenizer.eos_token_id] for tokens_seq in tokens['input_ids']]
            input_ids = torch.tensor(input_ids, device=model.device)
            attention_mask = torch.ones_like(input_ids)
            del tokens
            
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            del input_ids, attention_mask

            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.detach().cpu()})

            progress_bar.set_description(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
            torch.cuda.empty_cache()
        
        model.save_pretrained(f"checkpoint_after_epoch_{epoch}")
        scheduler.step()

    wandb.finish()

    # Save the fine-tuned model
    # model.save_pretrained("fine_tuned_model")

    return model