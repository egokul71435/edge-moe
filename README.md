conda environment -> edge_moe
uses python=3.12
datasets
tqdm


The only axes you vary:

Framework: torch-mps vs mlx
Model type: dense vs moe
Model scale: small (~125M) / medium (~350M) / large (~750M)