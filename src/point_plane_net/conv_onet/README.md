# Scene Reconstruction with Implicit Function

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `conv_onet` using
```
conda env create -f environment.yaml
conda activate conv_onet
```
**Note**: if the `pytorch_scatter` is not working, you can install it following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140).

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

To compile the dmc extension, you have to have a cuda enabled device set up.
If you experience any errors, you can simply comment out the `dmc_*` dependencies in `setup.py`.
You should then also comment out the `dmc` imports in `im2mesh/config.py`.

## Dataset

To evaluate a pretrained model or train a new model from scratch, you have to obtain the dataset.
To this end, there are two options:

1. you can download our preprocessed data
2. you can download the ShapeNet dataset and run the preprocessing pipeline yourself

Take in mind that running the preprocessing pipeline yourself requires a substantial amount time and space on your hard drive.
Unless you want to apply our method to a new dataset, we therefore recommmend to use the first option.

### Preprocessed data
You can download our preprocessed data (73.4 GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/ShapeNet` folder.


### Building the dataset
Alternatively, you can also preprocess the dataset yourself.
To this end, you have to follow the following steps:
* download the [ShapeNet dataset v1](https://www.shapenet.org/) and put into `data/external/ShapeNet`. 
* download the [renderings and voxelizations](http://3d-r2n2.stanford.edu/) from Choy et al. 2016 and unpack them in `data/external/Choy2016` 
* build our modified version of [mesh-fusion](https://github.com/davidstutz/mesh-fusion) by following the instructions in the `external/mesh-fusion` folder

You are now ready to build the dataset:
```
cd scripts
bash dataset_shapenet/build.sh
``` 

This command will build the dataset in `data/ShapeNet.build`.
To install the dataset, run
```
bash dataset_shapenet/install.sh
```

If everything worked out, this will copy the dataset into `data/ShapeNet`.

**Note**: Alternatively, you can put the downloaded/created data somewhere else, and then generate a hyperlink `data` pointing to this `data` folder.

```
ln -s ${path-to-data}/data data
```


