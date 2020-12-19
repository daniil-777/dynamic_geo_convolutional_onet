# Visualisation Planes

Run the following command:

```
python normal_per_class.py 
```

You should create 13 config files for every object in the folder **configs/pointcloud/normals** For every file you should specify its own class. Also you should create **configs/default_plane_vis.yaml** file and specify there encoder_kwargs: **{n_channels: _, plane_param_file: _}** (number of learned planes and the name of the file where normals of learned planes would ba saved).

If you want to get normals of planes just for one object then you should comment all **subprocess.call(["python", "generate_plane_vis.py", "configs/pointcloud/normals/01.yaml"])** for exception one class in **normal_per_class.py**
