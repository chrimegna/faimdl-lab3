import wandb
import torch
def eval_step(epoch, model, val_loader, criterion):
  model.eval() # Il modello sa che siamo in modalità val
  running_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad(): # Durante la validation non siamo interessati al gradiente
    for batch_idx, (inputs, targets) in enumerate(val_loader):
      inputs, targets = inputs.cuda(), targets.cuda()
      # Nel train_step sono 5 i passi. Nell'eval step sono 2:
      # - Calcolo outputs;
      # - Calcolo Loss;
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      # Adesso calcoliamo i valori di interesse:
      running_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  val_loss = running_loss/len(val_loader)
  val_acc = 100.* correct/total
  wandb.log({
      "epoch": epoch,
      "val_loss": val_loss,
      "val_accuracy": val_acc
  })
  return val_loss, val_acc 
    
