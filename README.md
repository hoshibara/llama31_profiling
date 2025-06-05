# BKC

```bash
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

cd pytorch
pip uninstall torch -y
pip uninstall torch -y
pip uninstall torch -y
git reset --hard HEAD
git clean -ffdx
git submodule deinit -f .
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton
python setup.py develop
cd ..


pip uninstall torchvision torchaudio -y
pip uninstall torchvision torchaudio -y
pip uninstall torchvision torchaudio -y
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu --no-deps

pip install transformers
```