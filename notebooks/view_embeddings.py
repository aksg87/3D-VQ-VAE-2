# %%
import pickle
from pathlib import Path

import lmdb
import numpy as np

# https://realpython.com/storing-images-in-python/#reading-from-lmdb

embeddings_dir = Path(
    "/myfilestore/efs_backups/akshay/vae_test_data/vae_output/"
    "version_1_epoch=1293-step=2587.lmdb"
    )

# %%

lmdb_env = lmdb.open(str(embeddings_dir), readonly=True) # for async set lock=False

# Start a new read transaction
with lmdb_env.begin() as txn:
    # Encode the key the same way as we stored it
    data = txn.get(str(0).encode())
    unpicked : np.ndarray = pickle.loads(data) # type: ignore
    print(unpicked[0].shape)
    print(unpicked[1].shape)
    print(unpicked[2].shape)

lmdb_env.close()
