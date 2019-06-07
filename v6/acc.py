
import numpy as np

lel = np.load('vgg64_lel.npy').item()
print (lel['train_acc'])
print (lel['val_acc'])

bp = np.load('vgg64_bp.npy').item()
print (bp['train_acc'])
print (bp['val_acc'])
