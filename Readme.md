# ResNet50 (Classification)

## Source of the model

	  Model picked up from 'https://github.com/quic/aimet-model-zoo/tree/develop/aimet_zoo_torch/resnet'

---

## Description of the model

	  > Model : ResNet50
	  > Input size:  (1,224,224,3)

---

## Framework and version

    AIMET   : torch-gpu-1.24.0
    offset  : 11
    pytorch : 1.9.1+cu111
    python  : 3.8

---

## Modifications done to the model (if any)

	
---

## Execution command
Command to run:
```bash
python src/resnet_quanteval.py --config config/resnet50_w8a8.json 
```

---

## list of operators in this model

 	{'Add', 'Conv', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Relu'}

---

## Trained on dataset(s)

Imagenet Pretrained


## Path to datasets

	- Internal Datsets - Used 2k images for calibration and 50k images for validation from Imagenet Dataset.
	- External Datasets - URLs

---

## Result:
  - resenet50_baseline
    - Top1: 76.060% 
    - Top5: 92.974%
    
  - resnet50_w8a8
    - Top1: 75.580% 
    - Top5: 92.762%

  - resnet50_w8a16
    - Top1: 75.742% 
    - Top5: 92.842%
  
---