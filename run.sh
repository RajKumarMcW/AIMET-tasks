#!/bin/sh -x

MODEL_FOLDER=aimet_model_zoo/Resnet50


VENV_NAME=venv_py38

python3.8 -m venv $VENV_NAME

echo "##################################"
echo "Creating the Environment Completed..."
echo "##################################"

sleep 1
echo "##################################"	
echo "Installing baseline Packages..."
echo "##################################"

$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install --upgrade pip
$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install -r $WORKSPACE/$MODEL_FOLDER/requirements.txt
$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

echo "##################################"
echo "Installing AIMET Packages..."
echo "##################################"

$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install https://github.com/quic/aimet/releases/download/1.24.0/AimetCommon-torch_gpu_1.24.0-cp36-cp36m-linux_x86_64.whl
$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install https://github.com/quic/aimet/releases/download/1.24.0/AimetTorch-torch_gpu_1.24.0-cp36-cp36m-linux_x86_64.whl -f https://download.pytorch.org/whl/torch_stable.html
$WORKSPACE/$VENV_NAME/bin/python3.8 -m pip install https://github.com/quic/aimet/releases/download/1.24.0/Aimet-torch_gpu_1.24.0-cp36-cp36m-linux_x86_64.whl


cat $WORKSPACE/$VENV_NAME/lib/python3.8/site-packages/aimet_common/bin/reqs_deb_common.txt |sudo xargs apt-get --assume-yes install
cat $WORKSPACE/$VENV_NAME/lib/python3.8/site-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt |sudo xargs apt-get --assume-yes install

source $WORKSPACE/$VENV_NAME/lib/python3.8/site-packages/aimet_common/bin/envsetup.sh
export LD_LIBRARY_PATH=$WORKSPACE/$VENV_NAME/lib/python3.8/site-packages/aimet_common:$LD_LIBRARY_PATH

echo "##################################"
echo "Environment Setting up Completed..."
echo "##################################"



echo "##################################"
echo "Running the baseline with dummy Input"
echo "##################################"
$WORKSPACE/$VENV_NAME/bin/python3.8 $WORKSPACE/$MODEL_FOLDER/src/resnet_quanteval.py --config $WORKSPACE/$MODEL_FOLDER/config/resnet50.json 