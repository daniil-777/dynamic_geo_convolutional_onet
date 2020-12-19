## (branch fc_plane_prediction)
## Using generate.py
For generating rotated objects, **use explicitly and only generate_rotation.py, instead of generate.py.** To compare differences between two, call:

`diff generate.py generate_rotation.py `


Moreover, there are updates in **im2mesh/onet/generation.py**, that support rotation of points in both input object and query points, that don't hurt your code if you want to train without any rotation using regular **generate.py **script. 


Before calling **generate_rotation.py**, it is mandatory and convenient to make your own config file and run it when calling for both **generate.py** and **eval_meshes.py**.
Example for your file should look like this:

```
inherit_from: configs/pointcloud/dpoc.yaml
test:
  model_file: /home/dsvilarkovic/partition600/occupancy_networks/out/pointcloud/dpoc/7planes_best.pt
model : 
  encoder_kwargs:
    hidden_dim: 32
    n_channels: 7
generation:
  generation_dir: generation_7plane_45_degrees
degrees: 45
```

Where you explicitly define **model.encoder_kwargs.n_channels** for number of planes, **generation.generation_dir** for specifying the name of the folder you want to save your reconstructions too (in this case, models will be saved in **out/pointcloud/dpoc/generation_7plane_45_degrees**), and **degrees** for rotation range you wish to include. Of course **test.model_file** path also needs to be changed in order if you want to generate with your own model.

There can be problems with matpltlib in generate_rotation.py, you may comment it
Finally, call:

`python generate_rotation.py configs/pointcloud/CONFIG_NAME.yaml --no-cuda`

## Using eval_meshes.py
For evaluating generated object, you have two options:
* reconstructed rotated object vs rotated object
* reconstructed rotated object vs non-rotated object

**reconstructed rotated object vs non-rotated object** is the mode that gave us the best result in the 7 plane DPCO rotation invariance test.** For that you don't need to change anything in the code**, just call eval_meshes.py with the configuration you defined for generate.py. Make sure you call correct version of **fc_plane_net.py**, I might have left canonical planes try out.

In this case, call: 

`python eval_meshes.py configs/pointcloud/CONFIG_NAME.yaml --no-cuda`


[recommended]**reconstructed rotated object vs rotated object** is the mode we expected to work better, since we expected rotated reconstruction. It will rotate the output object for which our reconstructed rotated object is evaluated to. For this one, in calling eval_meshes, add argument --eval-rotations

In this case, call:

`python eval_meshes.py configs/pointcloud/CONFIG_NAME.yaml --no-cuda --eval-rotations`


## Training with rotation augmentation by using train_point_plane_rotate.py

In order to rotate objects in training we use this code which depends on:
* im2mesh/onet/training.py 
* im2mesh/onet/generation.py

To run rotation augmentation training make config file:


```
inherit_from: configs/pointcloud/dpoc.yaml
model : 
  encoder_kwargs:
    hidden_dim: 32
    n_channels: 7
generation:
  generation_dir: generation_7plane_45_degrees_rotation_augmentation
degrees: 45
```

`python train_point_plane_rotate.py configs/pointcloud/CONFIG_NAME.yaml MODEL_NAME`

Where you explicitly define **model.encoder_kwargs.n_channels** for number of planes, **generation.generation_dir** for specifying the name of the folder you want to save your logs and models (in this case, models will be saved in **out/pointcloud/dpoc/generation_7plane_45_degrees_rotation_augmentation**), and **degrees** for rotation range you wish to include for rotation augmentation. In this case as opposed to **generate_rotate.py** there is no need for **test.model_file** since you are making one with training.

## About rotations

Rotation ranges are defined by your configuration, and exact value for rotation per each axis are chosen by random in range 0-degrees (in this case 0-45 degrees) for each axis (so you might get 32 degrees for x-rotation, 12 degrees for y-rotation and 1 degrees for z-rotation). Rotations are randomly chosen for each object, and implementation can be found in rotate_points function in **im2mesh/onet/generation.py** file. 
Moreover, rotation tuple in form **(x_angle, y_angle, z_angle)** is kept (in the file under name **{object_code}_rotation.pckl**) on the same path where the reconstructed object is.

## About generating image graph of rotated, non-rotated and reconstructed rotated object
In **generate_rotated_mesh** function in **im2mesh/onet/generation.py** file, there is plotting of files on random given by my silly counter (random_counter = int(random.random()* 10000000000000 % 256) to generate matplotlib plots for some of the objects being reconstructed. If you want every object to be reconstructed, change 256 to 1.


## rotate_points function
To understand what is going on im2mesh/onet/generation.py, I will give you couple of explanation steps:

On line 239, there is a : 

`pointsf = self.rotate_points(pointsf,query_points = True)`

This function is used to rotate query points.    

On line 116, there is a : 

`inputs, rotation = self.rotate_points(inputs, DEGREES = DEGREES)`

This function rotates object pointcloud that will be put in the encoder.
