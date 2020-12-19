from os.path import join, exists, isdir
from os import listdir
import subprocess
from shutil import move
from tqdm import tqdm


ds_path = '/is/sg/mniemeyer/Documents/Development/scalable_onet/code/data/RoomDataset5'
scene_path = '/is/sg/mniemeyer/Documents/Development/scalable_onet/code/data/scenes5'
scene_ext = 'obj'
#CMD = '/is/sg/mniemeyer/Documents/Development/scalable_onet/code/external/binvox -c -d 64 %s'
#CMD = 'xvfb-run -s "-screen 0 640x480x24" ./binvox -c -d 64 %s'
CMD64 = 'bash voxelize.sh 64 %s'
CMD32 = 'bash voxelize.sh 32 %s'

classes = [cl for cl in listdir(ds_path) if isdir(join(ds_path, cl))]
classes.sort()

for cl in classes:
    print("Processing class %s ..." % cl)
    cl_path = join(ds_path, cl)
    cl_path_scene = join(scene_path, cl)
    models = [m for m in listdir(cl_path) if isdir(join(cl_path, m))]
    models.sort()
    scenes = [m for m in listdir(cl_path_scene) if m[-3:] == scene_ext]
    scenes.sort()
    assert(len(models) == len(scenes))

    for idx, model in tqdm(enumerate(models)):
        scene_name = scenes[idx]
        scene_file = join(cl_path_scene, scene_name)
        binvox_file = '%s.binvox' % scene_file[:-4]

        try:
            binvox_file_out = join(cl_path, model, 'model64.binvox')
            if not exists(binvox_file_out):
                CMD_i = (CMD64 % scene_file).split()
                cmd = subprocess.Popen(CMD_i)
                cmd.communicate()
                cmd.wait()
                move(binvox_file, binvox_file_out)

            binvox_file_out = join(cl_path, model, 'model32.binvox')
            if not exists(binvox_file_out):
                CMD_i = (CMD32 % scene_file).split()
                cmd = subprocess.Popen(CMD_i)
                cmd.communicate()
                cmd.wait()
                move(binvox_file, binvox_file_out)
        except Exception as e:
            print("error for", model)
