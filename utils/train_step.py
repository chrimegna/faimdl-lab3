import wandb
def train_step(epoch, model, train_loader, criterion, optimizer):
  model.train() # Il modello sa che siamo in modalità training
  running_loss = 0.0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    # Eseguiamo i 5 passi del training:
    # - azzerare il gradiente;
    # - determinare gli output;
    # - calcolare la loss;
    # - calcolare l'influenza di ciascun neurone sulla loss;
    # - aggiornare i pesi
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    # Adesso calcoliamo i valori di interesse:
    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
  train_loss = running_loss/len(train_loader)
  train_acc = 100.* correct/total
  wandb.log({
      "epoch": epoch,
      "train_loss": train_loss,
      "train_accuracy": train_acc
  }) 
  return train_loss, train_acc
    
