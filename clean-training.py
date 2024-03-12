import shutil
try:
    shutil.rmtree("model_lstm")
    shutil.rmtree("model_bert")
except FileNotFoundError:
    print("ERROR: The folders 'model_lstm' or 'model_bert' were not found.")
    print("Please finish training first before cleaning.")
    exit(1)
print("Cleaned training checkpoints")
