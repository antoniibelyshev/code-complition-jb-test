import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model,
    tokenizer,
    train_dataset,
    batch_size = 2,
    epochs = 1,
    learning_rate = 5e-5,
):
    train_loader = DataLoader(torch.arange(len(train_dataset)), batch_size=batch_size, shuffle=True)

    # freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    for param in model.model.layers[-1].parameters():
        param.requires_grad = True

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Fine-tuning loop
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for batch in progress_bar:
            tokens = tokenizer([train_dataset[i] for i in batch], padding=True, truncation=True)
            del batch

            input_ids = torch.tensor(tokens['input_ids'], device=model.device)
            attention_mask = torch.tensor(tokens['attention_mask'], device=model.device)
            del tokens
            
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            del input_ids, attention_mask

            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
            torch.cuda.empty_cache()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")

    return model