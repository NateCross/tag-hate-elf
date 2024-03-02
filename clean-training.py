import shutil
try:
    shutil.rmtree("train_lstm")
    shutil.rmtree("train_bert")
except FileNotFoundError:
    print("ERROR: The folders 'train_lstm' or 'train_bert' were not found.")
    print("Please finish training first before cleaning.")
    exit(1)
print("Cleaned training checkpoints")
