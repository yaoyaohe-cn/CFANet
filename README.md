# CFANet
A Channel Frequency Adaptive Network (CFANet) for long-term time series forecasting.

## Get started
Follow these steps to get started with CFANet:
### 1. Install Requirements
Install Python 3.8 and the necessary dependencies.

```bash
pip install -r requirements.txt
```
### 2. Download Data
All the datasets needed for CFANet can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. 

### 3. Train the model
We provide the experiment scripts of all benchmarks under the folder ```./scripts/``` to reproduce the results.

```
bash scripts/ettm1.sh
bash scripts/ettm2.sh
bash scripts/etth1.sh
bash scripts/etth2.sh
bash scripts/electricity.sh
bash scripts/traffic.sh
bash scripts/weather.sh
```
