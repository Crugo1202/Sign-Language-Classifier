import os
import shutil

# Copy train_topk_official.py to train_expA.py, train_expB.py, train_expC.py
for exp in ['A', 'B', 'C']:
    shutil.copy('train_topk_official.py', f'train_exp{exp}.py')

