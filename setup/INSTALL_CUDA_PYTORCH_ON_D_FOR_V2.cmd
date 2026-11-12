@echo off
setlocal
rem EndekGAN-T2I V2: install CUDA packages ONLY in existing isolated D: research venv
set "PY=D:\VSCode-MultiAgent-AI\envs\endekgan-t2i-v2-py311\Scripts\python.exe"
set "PIP_CACHE_DIR=D:\VSCode-MultiAgent-AI\caches\pip"
set "TEMP=D:\VSCode-MultiAgent-AI\caches\temp"
set "TMP=D:\VSCode-MultiAgent-AI\caches\temp"
set "MPLCONFIGDIR=D:\VSCode-MultiAgent-AI\caches\matplotlib"
set "TORCH_HOME=D:\VSCode-MultiAgent-AI\caches\torch"
set "HF_HOME=D:\DISERTASI_S3\WIWIK\DATASET_L2\EndekGAN\EndekGAN-T2I\MODEL_CACHE\huggingface"
set "IPYTHONDIR=D:\VSCode-MultiAgent-AI\ipython"
set "JUPYTER_DATA_DIR=D:\VSCode-MultiAgent-AI\jupyter\data"
set "JUPYTER_RUNTIME_DIR=D:\VSCode-MultiAgent-AI\jupyter\runtime"
set "JUPYTER_CONFIG_DIR=D:\VSCode-MultiAgent-AI\jupyter\config"
set "PYTHONPYCACHEPREFIX=D:\VSCode-MultiAgent-AI\caches\pycache-endek"
if not exist "%PY%" (
  echo Cannot find D: environment: %PY%
  exit /b 1
)
"%PY%" -c "import sys; assert sys.prefix.lower().startswith(r'd:\vscode-multiagent-ai'); assert sys.base_prefix.lower().startswith(r'd:\vscode-multiagent-ai'); print(sys.executable, sys.base_prefix)"
if errorlevel 1 exit /b 1
echo.
echo Pinning scientific dependencies compatible with the old EndekGAN PyTorch 2.3.1 code.
"%PY%" -m pip install "numpy==1.26.4" "pandas==2.2.3" "Pillow==10.4.0" "matplotlib==3.8.4"
if errorlevel 1 exit /b 1
echo Installing PyTorch 2.3.1 CUDA 12.1 for RTX 3070 Ti in isolated D: venv.
echo GPU driver newer than CUDA 12.1 can run cu121 wheels.
"%PY%" -m pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 1
"%PY%" -m pip install "scikit-learn==1.5.2" "transformers==4.44.2" "torchmetrics==1.4.2" "torch-fidelity==0.3.0" "lpips==0.1.4" "sentencepiece==0.2.0"
if errorlevel 1 exit /b 1
"%PY%" -c "import torch,torchvision,sklearn,transformers,torchmetrics,torch_fidelity,lpips; print('PyTorch',torch.__version__);print('CUDA build',torch.version.cuda);print('CUDA available',torch.cuda.is_available()); print('Python location',__import__('sys').executable); assert torch.version.cuda is not None"
if errorlevel 1 exit /b 1
echo.
echo CUDA PYTORCH INSTALLED ON D. Check free VRAM before running any full GAN training.
echo Use 05_EndekGAN_T2I_Train_Preprocessed_Crop_V2.ipynb
endlocal
