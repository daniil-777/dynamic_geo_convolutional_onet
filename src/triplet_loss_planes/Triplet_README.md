## Training Dynamic Plane Convolutional Occupancy Network With Triplet Loss
```
python train_point_plane.py triplet_fc_plane.yaml model_name
```

You should specify ``margin_triplet`` in ``training`` part. Choose ``triplet_fc_plane_net`` as class model in ``encoder``, ``cbatchnorm`` in ``decoder``