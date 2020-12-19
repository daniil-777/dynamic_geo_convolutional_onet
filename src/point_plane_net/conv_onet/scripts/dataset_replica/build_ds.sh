obj_file=/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/replica_processed/scenes/room_0/mesh.ply
RESOLUTION=256
NVIEWS=64
out_folder=/is/rg/avg/mniemeyer/project_data/2020/scalable_onet/data/Replica_test2
BLENDER='/is/sg/mniemeyer/Bin/blender-2.79b-linux-glibc219-x86_64/blender'

$BLENDER --background --python render_script.py -- $obj_file --resolution $RESOLUTION --output_folder $out_folder --views $NVIEWS --camera 'inside'