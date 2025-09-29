conda activate gymnasium_env
cd /home/yvonne/wm/Gymnasium-Robotics
python data_gen_antmaze.py

pip install gymnasium-robotics
pip install minari
pip install "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" --index-url https://download.pytorch.org/whl/cu121
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
pip install av==10.0.0