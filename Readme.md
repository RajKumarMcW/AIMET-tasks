# ResNet (Classification)

## Usage
To run evaluation with QuantSim in AIMET, use the following
```bash
bash run.sh
```
In run.sh
```bash
python resnet_quanteval.py\
  --model-config <configuration to be tested> \
  --dataset-path <path to ImageNet dataset> \
  --use-cuda <whether to run on GPU or cpu(TRUE/FALSE)>
```

Available model configurations are:
- resnet50_w8a8
- resnet50_w8a16
---

Result:
- resnet50_w8a8
    - FP32 accuracy     : 76.060%
    - Quantized accuracy: 75.896%
- resnet50_w8a16
    - FP32 accuracy     : 76.060%
    - Quantized accuracy: 75.966%