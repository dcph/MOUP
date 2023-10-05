base
bash Anaconda3-xxxx-Linux-x86_64.sh
conda create -n d4rl_env python=3.7 

mujuco
git clone https://github.com/google-deepmind/mujoco.git
mv mujoco210-linux-x86_64.tar.gz mujoco210
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
cp -r mujoco210 ~/.mujoco
sudo gedit ~/.bashrc   
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
source ~/.bashrc  
pip install mujoco_py
pip install dm_control

d4rl
git clone https://github.com/rail-berkeley/d4rl.git
pip install -e .

moup
code and environment for ours is like https://github.com/yihaosun1124/OfflineRL-Kit.git


git clone https://github.com/dcph/MOUP.git
python run_moup.py


