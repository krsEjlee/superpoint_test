# SuperPoint

## 개요

본 리포지토리는 "SuperPoint: Self-Supervised Interest Point Detection and Description" (저자: Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, ArXiv 2018)의 TensorFlow 구현체입니다. 또한 PyTorch 버전(MIT 라이센스)도 제공되며, [superpoint_pytorch.py](https://github.com/rpautrat/SuperPoint/blob/master/superpoint_pytorch.py)에서 확인할 수 있습니다.
이 코드는 기존 오픈소스 파일을 수정하여 파인튜닝 및 모델별 성능 평가(예: engine 데이터셋)를 진행하기 위한 목적으로 사용됩니다.

## 주요 구성 요소

- **코드 및 설정 파일:**
  - 주요 학습 및 추론 스크립트는 `superpoint/` 폴더에 위치합니다.
  - 설정 파일들은 `superpoint/configs/` 폴더에 저장되어 있습니다.
- **데이터셋:**
  - MS-COCO (train, val), HPatches 데이터셋을 다운로드해야 하며, Synthetic Shapes 데이터셋은 자동으로 생성됩니다.
- **평가 노트북:**
  - HPatches를 활용한 detector repeatability 및 descriptor 평가를 위한 노트북들이 `notebooks/` 폴더에 있습니다.
- **사전 학습 모델:**
  - MagicPoint 및 SuperPoint의 사전 학습 가중치(예: `sp_v6`)가 제공됩니다. 사전 학습 모델은 아래의 지침에 따라 다운로드 및 압축 해제합니다.

## 시스템 요구사항

- Python 3.6.1
- CUDA 11.x 및 cuDNN 8.0 (Ampere GPU 지원)
- TensorFlow GPU 버전 (tensorflow-gpu==2.4.1, Ampere GPU 지원)
- 기타 패키지: `requirement.txt`에 명시된 패키지와 tensorflow-addons

## 설치 방법

```bash
# 1. 가상환경 생성 및 활성화
conda create -n superpoint_ampere python=3.6
conda activate superpoint_ampere

# 2. CUDA 11.x와 cuDNN 8 설치
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.0

# 3. TensorFlow GPU 버전 설치
pip install tensorflow-gpu==2.4.1

# 4. Python 의존성 패키지 설치
pip install -r requirements.txt
pip install tensorflow-addons

# 5. 소스 코드 개발 모드로 설치
pip install -e .

# 6. 추가 셋업 스크립트 실행
sh setup.sh
# 7. CUDA 사용 장치 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 8. GPU 사용 현황 모니터링 (선택 사항)
watch -d -n 0.5 nvidia-smi
```

## 데이터셋 준비

데이터셋 디렉토리(예: `$DATA_DIR`)를 생성하고 아래와 같은 폴더 구조로 구성합니다:

```
$DATA_DIR
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   -- ...
|   -- val2014
|       |-- file1.jpg
|       -- ...
|-- HPatches
|   |-- i_ajuntament
|   -- ...
`-- synthetic_shapes (첫 실행 시 자동 생성)
```

- [MS-COCO 2014](http://cocodataset.org/#download)와 [HPatches](https://www.kaggle.com/api/v1/datasets/download/javidtheimmortal/hpatches-sequence-release)를 `$DATA_DIR`에 다운로드하세요.

## 사용 방법 및 워크플로우

**참고:** 모든 명령은 `superpoint/` 하위 폴더 내에서 실행해야 합니다.

1. **Synthetic Shapes를 활용한 MagicPoint 학습**

   ```bash
   python experiment.py train superpoint/configs/magic-point_shapes.yaml magic-point_synth
   python superpoint/experiment.py train superpoint/configs/magic-point_shapes.yaml magic-point_synth
   ```

   - 학습 결과(가중치 및 TensorBoard 로그)는 `$EXPER_DIR/magic-point_synth/`에 저장됩니다.

2. **MS-COCO에서 추론 결과(Detection) Export**

   ```bash
   python superpoint/export_detections.py superpoint/configs/magic-point_coco_export.yaml magic-point_synth --pred_only --batch_size=5 --export_name=magic-point_coco-export1
   python superpoint/export_detections.py superpoint/configs/magic-point_coco_export.yaml magic-point_synth --pred_only --batch_size=5 --export_name=magic-point_hpatches-export1
      python superpoint/export_detections.py superpoint/configs/magic-point_hannarae_export.yaml magic-point_synth --pred_only --batch_size=5 --export_name=magic-point_hannarae_export
   ```

   - 결과는 `$EXPER_DIR/outputs/magic-point_coco-export1/`에 저장됩니다.
   - 결과는 `$EXPER_DIR/outputs/magic-point_hannarae-engine/`에 저장됩니다.

3. **MS-COCO에서 MagicPoint 재학습**

   ```bash
   python superpoint/experiment.py train superpoint/configs/magic-point_coco_train.yaml magic-point_coco2

   python superpoint/experiment.py train superpoint/configs/magic-point_hannarae_train.yaml magic-point_hannarae_2
   ```

   - `magic-point_coco_train.yaml` 파일에서 interest point 라벨 경로(`data/labels` 항목)를 설정한 후 실행합니다.

4. **HPatches를 활용한 Detector Repeatability 평가**

**참고:** 모델 평가시 GPU한개만 사용함
export CUDA_VISIBLE_DEVICES=0

```bash
python superpoint/export_detections_repeatability.py superpoint/configs/magic-point_repeatability.yaml magic-point_coco2 --export_name=magic-point_hpatches-repeatability-v-coco2

python superpoint/export_detections_repeatability.py superpoint/configs/magic-point_repeatability.yaml magic-point_hannarae_2 --export_name=magic-point_hpatches-repeatability-hannarae2

```

- 평가 시, `data/alteration` 항목을 통해 viewpoint 또는 illumination 변화를 선택합니다.
- 이후 `notebooks/detector_repeatability_hpatches.ipynb` 노트북에서 평가를 진행합니다.
- 기존의 classical detector 평가도 `classical-detectors_repeatability.yaml` 설정 파일을 사용해 진행할 수 있습니다.

5. **MS-COCO에서 SuperPoint 학습**

   ```bash
   python superpoint/experiment.py train superpoint/configs/superpoint_coco.yaml superpoint_coco
   ```

   - 여러 번의 homographic adaptation (보통 1~2회) 후에 SuperPoint 학습을 진행합니다.

6. **HPatches를 활용한 Descriptor 평가 (Homography Estimation)**

   ```bash
   python superpoint/export_descriptors.py superpoint/configs/superpoint_hpatches.yaml superpoint_coco --export_name=superpoint_hpatches-v

   python superpoint/export_descriptors.py superpoint/configs/superpoint_hpatches_hannarae.yaml superpoint_coco --export_name=superpoint_hannarae_hpatches-v

   ```

   - 평가 시, 역시 `data/alteration` 항목을 통해 viewpoint 또는 illumination 변화를 선택합니다.
   - 이후 `notebooks/descriptors_evaluation_on_hpatches.ipynb` 노트북을 통해 평가를 진행합니다.
   - classical descriptor 평가도 `classical-descriptors.yaml` 설정 파일로 수행할 수 있습니다.

7. **사전 학습 가중치를 활용한 매칭 데모 (Matching Features Demo)**

   ```bash
   tar -xzvf pretrained_models/sp_v6.tgz $EXPER_DIR/saved_models/sp_v6
   ```

   - `sp_v6` 라벨이 붙은 사전 학습 가중치를 다운로드 받아 `$EXPER_DIR/saved_models/sp_v6` 경로에 압축 해제합니다.
   - 데모 실행:

   ```bash
   python match_features_demo.py sp_v6 $DATA_PATH/HPatches/i_pool/1.ppm $DATA_PATH/i_pool/6.ppm
   ```

   - 추가 옵션(`--H`, `--W`, `--k_best`)를 통해 이미지 크기 조정 및 최대 keypoint 수 설정이 가능합니다.

8. **사전 학습 모델 파인튜닝**
   ```bash
   python superpoint/experiment.py train superpoint/configs/superpoint_coco.yaml superpoint_finetuned --pretrained_model sp_v6
   ```
   - 사전 학습 모델 다운로드 후, 예를 들어 SuperPoint 파인튜닝을 진행하려면 위 명령어를 실행합니다.
   - 또는, 기존에 학습된 `superpoint_coco` 체크포인트에서 파인튜닝:
   ```bash

   ```

````

saved model
python superpoint/export_model.py superpoint/configs/superpoint_hannarae.yaml superpoint_coco


## 추가 코드 수정 사항
아래의 수정 사항들을 적용하여 코드가 원활하게 동작하도록 합니다.

1. **experiment.py 파일의 sys.path 수정**
 `experiment.py` 파일 상단에 아래 코드를 추가합니다:
 ```python
 import sys
 sys.path.append(os.path.abspath(os.path.dirname(__file__)))
 sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
````

2. **datasets/synthetic_shapes.py 수정 (72, 132번째 줄)**  
   임시 디렉토리 설정을 아래와 같이 수정합니다:

   ```python
   temp_dir = Path(os.environ.get('TMPDIR', '/tmp'), primitive)
   ```

3. **TensorFlow 호환성 수정**  
   기존의 아래 코드:

   ```python
   import tensorflow as tf
   ```

   를 아래와 같이 변경:

   ```python
   import tensorflow.compat.v1 as tf
   tf.disable_v2_behavior()
   ```

   또한, 더 이상 사용되지 않는 모듈:

   ```python
   from tensorflow.contrib.image import transform as H_transform
   ```

   를 아래와 같이 수정:

   ```python
   import tensorflow_addons as tfa
   H_transform = tfa.image.transform
   ```

## 기타 참고 사항

**입력 이미지 크기:**  
MagicPoint와 SuperPoint는 이미지 크기가 8의 배수여야 합니다. 설정 파일의 data->preprocessing->resize 옵션 등을 활용하여 이미지 크기를 조정하세요.
