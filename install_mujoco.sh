#!/bin/sh

# 1. Make sure Cython<3 is installed.
# 2. Run the following command.

MUJOCO_INSTALL_DIR=~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_INSTALL_DIR/mujoco210/bin:/usr/lib/nvidia

# 3. Install Mujoco to your local folder with the following.
mkdir -p $MUJOCO_INSTALL_DIR && \
wget https://www.tdmpc2.com/files/mjkey.txt && \
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
tar -xzf mujoco210-linux-x86_64.tar.gz && \
rm mujoco210-linux-x86_64.tar.gz && \
mv mujoco210 $MUJOCO_INSTALL_DIR/mujoco210 && \
mv mjkey.txt $MUJOCO_INSTALL_DIR/mjkey.txt

# 4. Trigger a compilation of Mujoco by importing `mujoco_py`.
export MUJOCO_GL=egl
python -c "import mujoco_py"

# Now, in .bashrc, update the LD_LIBRARY_PATH variable.
