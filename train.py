import torch
import wandb
import os

# Ora importiamo i nostri moduli

from data.data_setup import setup
from dataset import dataset_dataloader as set_load 
from models import Model1
from utils.eval_step import eval_step
from utils.train_step import train_step

def run_training(model, epochs, batch_size, learning_rate):
  wandb.init(
      project = "Tiny-imagenet-project",
      config={
          "Learning rate": learning_rate,
          "Batch size": batch_size,
          "Epochs": epochs,
          "Architecture": model
      }
  )
  model.to("cuda")
  data_path = "data"
  if not os.path.exists(os.path.join(data_path, "tiny-imagenet-200")):
      setup(data_path)
    
  # Creiamo i dataset e i loader
  train_dir = os.path.join(data_path, "tiny-imagenet-200/train")
  val_dir = os.path.join(data_path, "tiny-imagenet-200/val")

  train_dataset, val_dataset = set_load.dataset(train_dir, val_dir)
  train_loader, val_loader = set_load.dataloader(train_dataset, val_dataset, batch_size)

  # 4. MODELLO, LOSS E OTTIMIZZATORE
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  # 5. IL CICLO DELLE EPOCHE (Quello che volevi salvare!)
  print(f"Inizio training...")
    
  for epoch in range(1, epochs+1):
    # Fase di Training
    train_loss, train_acc = train_step(epoch, model, train_loader, criterion, optimizer)
      
    # Fase di Validazione
    val_loss, val_acc = eval_step(epoch, model, val_loader, criterion)
        
    # Salvataggio checkpoint (opzionale ma consigliato)
    if (epoch + 1) % 5 == 0:
      checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
      os.makedirs("checkpoints", exist_ok=True)
      torch.save(model.state_dict(), checkpoint_path)
      print(f"Checkpoint salvato: {checkpoint_path}")

    # Chiudiamo la sessione WandB
  wandb.finish()
  print("Addestramento completato!")

if __name__ == "__main__":
    run_training(model, epochs, batch_size, learning_rate)
