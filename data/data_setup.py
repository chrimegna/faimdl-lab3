import os
import shutil
from io import BytesIO
import requests
import zipfile
def setup(destinazione):
  dataset_url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  print("Download in corso attendi...")
  response = requests.get(dataset_url)
  if response.status_code == 200:
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
      zip_file.extractall(destinazione)
      print("Download ed estrazione completati!")
  else:
    print(f"Errore nel download: {response.status_code}")
    return
  base_path = os.path.join(destinazione, "tiny-imagenet-200")
  val_dir = os.path.join(base_path, "val")
  annotations_file = os.path.join(val_dir, "val_annotations.txt")

  print("Sistemazione cartella Validation...")
  with open(annotations_file, "r") as f:
    for line in f:
      # Dividiamo la riga del file txt
      img_name, class_id, *_ = line.split("\t")
            
      # Creiamo la sottocartella della classe se non esiste
      class_dir = os.path.join(val_dir, class_id)
      os.makedirs(class_dir, exist_ok=True)
           
      # Copiamo l'immagine dalla cartella 'images' alla nuova cartella della classe
      src_path = os.path.join(val_dir, "images", img_name)
      dst_path = os.path.join(class_dir, img_name)
            
      if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)

    # Pulizia: rimuoviamo la cartella immagini originale che ora è vuota/inutile
    shutil.rmtree(os.path.join(val_dir, "images"))
    print("Procedura completata con successo!")

if __name__ == "__main__":
    setup(destinazione)
