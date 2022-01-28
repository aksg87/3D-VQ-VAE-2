python train.py "/myfilestore/efs_backups/akshay/vae_test_data/vae_test_data"

python plot_from_checkpoint.py "/myfilestore/efs_backups/akshay/vae_test_data/vae_test_data" "/home/akshay_goel_tempus_com/3D-VQ-VAE-2/vqvae/lightning_logs/version_1/checkpoints/epoch=1293-step=2587.ckpt" "/myfilestore/efs_backups/akshay/vae_test_data/vae_output/output.nrrd"