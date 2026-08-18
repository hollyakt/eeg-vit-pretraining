import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = 'logs/training_losses.csv'
if not os.path.exists(csv_path):
    raise SystemExit(f'Missing {csv_path}')

steps = []
losses = []
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        steps.append(int(row[0]))
        losses.append(float(row[1]))

plt.figure(figsize=(8,4))
plt.plot(steps, losses, label='train_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

import os
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/pretrain_loss.png')
print('Saved plot to plots/pretrain_loss.png')
