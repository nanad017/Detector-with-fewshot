# TRAINING GUIDE — 4 Malware Detection Projects on Custom Dataset

## Tổng quan

Train 4 project trên dataset 5 họ malware + 1 benign (~9,940 mẫu PE). Mỗi project có định dạng đầu vào và kiến trúc khác nhau.

| # | Project | Framework | Classification | Input Format | Output Model |
|---|---------|-----------|---------------|-------------|-------------|
| 1 | CNN (malware-classification-CNN) | PyTorch | **Family (6 lớp)** | .png ảnh xám 256×256 | `models/cnn/best_model.pt` |
| 2 | deep-malware-detection | PyTorch | **Binary** | .pickle (PE header bytes) | `models/deepmd/best_model.pt` |
| 3 | SOREL-FFNN (thay SOREL-20M) | PyTorch | **Binary** | EMBER v3 vector (2386-dim) | `models/sorel_ffnn/best_model.pt` |
| 4 | ember2024 LightGBM | LightGBM | **Binary + Family** | EMBER v3 vector (2386-dim) | `models/ember_lgbm/best_model.model` |

### Vì sao thay SOREL-20M bằng SOREL-FFNN?

SOREL-20M yêu cầu EMBER v2 features (2381-dim) đã được lưu trong LMDB + meta.db SQLite. Code trích xuất feature gốc là internal tool của Sophos, **không có trong repo công khai**. Để dựng lại pipeline này cho 10K mẫu:
- Cần cài EMBER gốc (Python 3.6 + LIEF 0.9.0) — xung đột với các project khác
- Cần tự xây database LMDB + SQLite đồng bộ — kỹ thuật phức tạp, không tỉ lệ với lượng dữ liệu

**Giải pháp thay thế**: Train một FFNN có kiến trúc giống `PENetwork` (multi-layer FC + binary head) nhưng dùng EMBER v3 features (2386-dim) từ thư viện `thrember` của ember2024 — code trích xuất feature đã có sẵn và được test.

---

## 1. Khả năng tương thích dataset

### 1.1 malware-classification-CNN — Tương thích (cần convert)

| Yêu cầu | Trạng thái |
|---|---|
| Định dạng | Cần .png ảnh xám, tổ chức theo thư mục lớp |
| Số lớp | 6 (5 họ + benign) — phù hợp |
| Số mẫu | ~9,940 — đủ cho CNN nhỏ |
| GPU | Không cần, CPU train được |

**Cần làm**: Convert .exe → .bytes → .png bằng script `convert_exe_to_png.py`.

### 1.2 deep-malware-detection — Tương thích (cần convert)

| Yêu cầu | Trạng thái |
|---|---|
| Định dạng | Cần .pickle chứa list[int] của PE header bytes |
| Số lớp | **Binary only** — gộp 5 họ malware thành 1 nhãn "malware" |
| Số mẫu | ~8,940 malware + 1,000 benign — chấp nhận được |
| Imbalance | Có (89% malware), sẽ dùng class weight |

**Cần làm**: Chạy `extract_header.py` (có sẵn trong repo) để convert .exe → .pickle.

### 1.3 ember2024 — Tương thích tốt

| Yêu cầu | Trạng thái |
|---|---|
| Định dạng | Dùng `PEFeatureExtractor.feature_vector()` trực tiếp từ raw bytes |
| Số lớp | Hỗ trợ cả binary và family |
| Số mẫu | 10K ít hơn nhiều so với 3.2M của EMBER2024 gốc, nhưng LightGBM hoạt động tốt với dữ liệu nhỏ |

**Cần làm**: Extract EMBER v3 feature vectors → train LightGBM.

### 1.4 SOREL-FFNN (thay thế) — Tương thích

| Yêu cầu | Trạng thái |
|---|---|
| Định dạng | Dùng chung EMBER v3 features với ember2024 |
| Số lớp | Binary only |
| Kiến trúc | Mô phỏng PENetwork: Linear → LayerNorm → ELU → Dropout → Linear(1) |

**Cần làm**: Dùng chung file features đã extract cho ember2024.

---

## 2. Các file cần sửa trong từng repo

### 2.1 malware-classification-CNN — Không cần sửa repo gốc

Repo gốc là Jupyter notebooks (Google Colab), không có script .py. Toàn bộ pipeline được viết mới dưới dạng script độc lập:

| Script | Chức năng |
|---|---|
| `scripts/convert_exe_to_png.py` | Convert .exe → .png (thay `utils/data_conversion.ipynb`) |
| `scripts/train_cnn.py` | Train CNN (thay `combined_classifier/*.ipynb`) |
| `scripts/eval_cnn.py` | Đánh giá (thay cell evaluation trong notebook) |

### 2.2 deep-malware-detection — Không cần sửa

Dùng nguyên bản `extract_header.py` và `train.py` từ repo. Chỉ cần gọi đúng tham số.

### 2.3 ember2024 — Không cần sửa repo gốc

Script mới dùng `thrember` package (cài từ repo) để extract feature. Việc train/eval dùng LightGBM trực tiếp.

### 2.4 SOREL-FFNN — Không liên quan repo gốc

Script mới hoàn toàn, không phụ thuộc vào SOREL-20M repo.

---

## 3. Hướng dẫn train từng project

### 3.0 Chuẩn bị môi trường chung

```bash
# Tạo virtual env
cd ~/An_solo/detector
python3 -m venv venv
source venv/bin/activate

# Cài PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài thư viện chung
pip install numpy pandas scikit-learn tqdm matplotlib pillow pefile

# Cài thrember (EMBER v3 feature extractor)
pip install -e ~/An_solo/detector/ember2024/

# Cài LightGBM
pip install lightgbm

# Kiểm tra
python -c "import torch; print(torch.__version__)"
python -c "import thrember; print('thrember OK')"
python -c "import lightgbm; print('lightgbm OK')"
```

### 3.1 Train CNN (malware-classification-CNN)

**Classification**: Family (6 classes: 5 malware families + benign)
**Train from scratch**: Có

```bash
# Bước 1: Convert .exe → .png
python scripts/convert_exe_to_png.py \
    --input_dir ~/An_solo/dataset/train \
    --output_dir ~/An_solo/detector/processed_data/png/train

python scripts/convert_exe_to_png.py \
    --input_dir ~/An_solo/dataset/test \
    --output_dir ~/An_solo/detector/processed_data/png/test

# Bước 2: Train
python scripts/train_cnn.py \
    --train_dir ~/An_solo/detector/processed_data/png/train \
    --test_dir ~/An_solo/detector/processed_data/png/test \
    --output_dir ~/An_solo/detector/models/cnn \
    --epochs 30 \
    --batch_size 32 \
    --image_size 256

# Output:
#   models/cnn/best_model.pt       — model state dict
#   models/cnn/training_history.json
```

**Thời gian ước tính (CPU)**: ~2-4 giờ cho 30 epochs

### 3.2 Train deep-malware-detection (MalConv+)

**Classification**: Binary (malware vs benign)
**Train from scratch**: Có

```bash
# Bước 1: Gộp tất cả malware families vào 1 thư mục
mkdir -p ~/An_solo/detector/processed_data/raw/train/malware
mkdir -p ~/An_solo/detector/processed_data/raw/train/benign
mkdir -p ~/An_solo/detector/processed_data/raw/test/malware
mkdir -p ~/An_solo/detector/processed_data/raw/test/benign

# Copy malware (tất cả họ)
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    cp ~/An_solo/dataset/train/malware/$family/* ~/An_solo/detector/processed_data/raw/train/malware/
    cp ~/An_solo/dataset/test/malware/$family/* ~/An_solo/detector/processed_data/raw/test/malware/
done

# Copy benign
cp ~/An_solo/dataset/train/benign/* ~/An_solo/detector/processed_data/raw/train/benign/
cp ~/An_solo/dataset/test/benign/* ~/An_solo/detector/processed_data/raw/test/benign/

# Bước 2: Extract PE header → .pickle
python -m src.bin.extract_header \
    --input_dir ~/An_solo/detector/processed_data/raw/train/malware \
    --output_dir ~/An_solo/detector/processed_data/pickle/train/malware

python -m src.bin.extract_header \
    --input_dir ~/An_solo/detector/processed_data/raw/train/benign \
    --output_dir ~/An_solo/detector/processed_data/pickle/train/benign

python -m src.bin.extract_header \
    --input_dir ~/An_solo/detector/processed_data/raw/test/malware \
    --output_dir ~/An_solo/detector/processed_data/pickle/test/malware

python -m src.bin.extract_header \
    --input_dir ~/An_solo/detector/processed_data/raw/test/benign \
    --output_dir ~/An_solo/detector/processed_data/pickle/test/benign

# Bước 3: Train (từ thư mục deep-malware-detection/src/deep_malware_detection/)
cd ~/An_solo/detector/deep-malware-detection/src/deep_malware_detection

python train.py \
    --device cpu \
    --model MalConvPlus \
    --benign_dir ~/An_solo/detector/processed_data/pickle/train/benign \
    --malware_dir ~/An_solo/detector/processed_data/pickle/train/malware \
    --batch_size 16 \
    --val_size 0.15 \
    --test_size 0.0 \
    --checkpoint_dir ~/An_solo/detector/models/deepmd \
    --tag best_model \
    --seed 42

cd ~/An_solo/detector
# Output: models/deepmd/best_model.pt
```

**Lưu ý**: `extract_header.py` cần được chạy từ thư mục gốc của `deep-malware-detection/` để Python tìm được module `src`. Nếu gặp lỗi import, thêm `PYTHONPATH`:

```bash
PYTHONPATH=~/An_solo/detector/deep-malware-detection python -m src.bin.extract_header ...
```

### 3.3 Train SOREL-FFNN (thay thế SOREL-20M)

**Classification**: Binary
**Train from scratch**: Có

```bash
# Bước 1: Extract EMBER v3 features (dùng chung với ember2024)
python scripts/extract_ember_features.py \
    --input_dir ~/An_solo/dataset/train \
    --output_dir ~/An_solo/detector/processed_data/ember_features \
    --mode binary

# Bước 2: Train FFNN
python scripts/train_sorel_ffnn.py \
    --train_features ~/An_solo/detector/processed_data/ember_features/X_train.npy \
    --train_labels ~/An_solo/detector/processed_data/ember_features/y_train.npy \
    --output_dir ~/An_solo/detector/models/sorel_ffnn \
    --epochs 50 \
    --batch_size 128 \
    --seed 42

# Output:
#   models/sorel_ffnn/best_model.pt    — model state dict
#   models/sorel_ffnn/history.json     — training history
```

**Thời gian ước tính (CPU)**: < 30 phút

### 3.4 Train ember2024 LightGBM

**Classification**: Cả binary và family
**Train from scratch**: Có

```bash
# Binary classification
python scripts/train_ember_lgbm.py \
    --train_features ~/An_solo/detector/processed_data/ember_features/X_train.npy \
    --train_labels ~/An_solo/detector/processed_data/ember_features/y_train_binary.npy \
    --output_dir ~/An_solo/detector/models/ember_lgbm \
    --model_name binary_model \
    --objective binary

# Family classification
python scripts/train_ember_lgbm.py \
    --train_features ~/An_solo/detector/processed_data/ember_features/X_train.npy \
    --train_labels ~/An_solo/detector/processed_data/ember_features/y_train_family.npy \
    --output_dir ~/An_solo/detector/models/ember_lgbm \
    --model_name family_model \
    --objective multiclass

# Output:
#   models/ember_lgbm/binary_model.model
#   models/ember_lgbm/family_model.model
```

**Thời gian ước tính (CPU)**: < 5 phút

---

## 4. Hướng dẫn evaluate / test

### 4.1 Evaluate CNN

```bash
python scripts/eval_cnn.py \
    --model_path ~/An_solo/detector/models/cnn/best_model.pt \
    --test_dir ~/An_solo/detector/processed_data/png/test \
    --output_dir ~/An_solo/detector/models/cnn \
    --image_size 256 \
    --batch_size 32

# Output:
#   models/cnn/results.json          — accuracy, per-class F1, confusion matrix
#   models/cnn/confusion_matrix.png
```

### 4.2 Evaluate deep-malware-detection

```bash
cd ~/An_solo/detector/deep-malware-detection/src/deep_malware_detection

# Evaluate trên test set
python -c "
import torch
from dataset import make_loader, MalwareDataset
from utils import get_accuracy, plot_confusion_matrix, plot_roc_curve, predict
from models import MalConvPlus
import json

device = torch.device('cpu')
test_dataset = MalwareDataset(
    '~/An_solo/detector/processed_data/pickle/test/benign',
    '~/An_solo/detector/processed_data/pickle/test/malware'
)
test_loader = make_loader(test_dataset, batch_size=16, shuffle=False)

model = MalConvPlus(8, 4096, 128, 32, 0.5).to(device)
model.load_state_dict(torch.load('~/An_solo/detector/models/deepmd/best_model.pt', map_location=device))

acc = get_accuracy(model, test_loader, device)
print(f'Accuracy: {acc:.2f}%')
"

cd ~/An_solo/detector
```

### 4.3 Evaluate SOREL-FFNN

```bash
python scripts/eval_sorel_ffnn.py \
    --model_path ~/An_solo/detector/models/sorel_ffnn/best_model.pt \
    --test_features ~/An_solo/detector/processed_data/ember_features/X_test.npy \
    --test_labels ~/An_solo/detector/processed_data/ember_features/y_test.npy \
    --output_dir ~/An_solo/detector/models/sorel_ffnn

# Output:
#   models/sorel_ffnn/results.json
```

### 4.4 Evaluate ember2024 LightGBM

```bash
# Binary model
python scripts/eval_ember_lgbm.py \
    --model_path ~/An_solo/detector/models/ember_lgbm/binary_model.model \
    --test_features ~/An_solo/detector/processed_data/ember_features/X_test.npy \
    --test_labels ~/An_solo/detector/processed_data/ember_features/y_test_binary.npy \
    --output_dir ~/An_solo/detector/models/ember_lgbm \
    --model_name binary_model

# Family model
python scripts/eval_ember_lgbm.py \
    --model_path ~/An_solo/detector/models/ember_lgbm/family_model.model \
    --test_features ~/An_solo/detector/processed_data/ember_features/X_test.npy \
    --test_labels ~/An_solo/detector/processed_data/ember_features/y_test_family.npy \
    --output_dir ~/An_solo/detector/models/ember_lgbm \
    --model_name family_model

# Output:
#   models/ember_lgbm/binary_model_results.json
#   models/ember_lgbm/family_model_results.json
```

---

## 5. Script train_all.sh

Xem `scripts/train_all.sh`. Chạy:

```bash
chmod +x scripts/train_all.sh
bash scripts/train_all.sh
```

Script tự động chạy tuần tự: data prep → feature extraction → train từng model.

---

## 6. Script test_all.sh

Xem `scripts/test_all.sh`. Chạy:

```bash
chmod +x scripts/test_all.sh
bash scripts/test_all.sh
```

Script chạy evaluation cho tất cả model đã train và in bảng so sánh kết quả.

---

## 7. Vị trí lưu model output

```
~/An_solo/detector/
├── models/
│   ├── cnn/
│   │   ├── best_model.pt           # CNN state dict
│   │   ├── training_history.json   # Loss/accuracy per epoch
│   │   ├── results.json            # Evaluation results
│   │   └── confusion_matrix.png
│   ├── deepmd/
│   │   └── best_model.pt           # MalConv+ state dict
│   ├── sorel_ffnn/
│   │   ├── best_model.pt           # FFNN state dict
│   │   ├── history.json            # Training history
│   │   └── results.json            # Evaluation results
│   └── ember_lgbm/
│       ├── binary_model.model      # LightGBM binary model
│       ├── family_model.model      # LightGBM family model
│       ├── binary_model_results.json
│       └── family_model_results.json
├── processed_data/
│   ├── raw/                        # Raw files (flattened malware + benign)
│   ├── png/                        # Converted PNG images
│   ├── pickle/                     # PE header pickle files
│   └── ember_features/             # EMBER v3 .npy feature files
└── scripts/
    ├── train_all.sh
    ├── test_all.sh
    └── *.py
```

---

## 8. Chuyển đổi dữ liệu khi không tương thích

### 8.1 .exe → .png (cho CNN)

Script `convert_exe_to_png.py`:
1. Đọc file .exe dưới dạng byte array
2. Ghi ra file .bytes (hex dump, 16 bytes/dòng)
3. Đọc .bytes, reshape thành mảng 2D, lưu thành ảnh PNG xám
4. Ảnh resize về 256×256

Các file không parse được sẽ được bỏ qua và log ra stderr.

### 8.2 .exe → .pickle (cho deep-malware-detection)

Dùng `extract_header.py` có sẵn trong `deep-malware-detection/src/bin/`:
- Dùng `pefile.PE(file_path)` để parse PE header
- Trích `file.header` (danh sách byte của MS-DOS + COFF + Optional header)
- Pickle danh sách này

File không parse được (PEFormatError) sẽ bị skip.

### 8.3 .exe → EMBER v3 feature vector (cho SOREL-FFNN + ember2024)

Script `extract_ember_features.py`:
- Dùng `thrember.features.PEFeatureExtractor` để trích 2386-dim vector
- Xử lý từng file, skip file lỗi
- Lưu thành `.npy` cho train và test

---

## 9. Lưu ý quan trọng

- **Tất cả script chạy trên CPU** — không yêu cầu GPU.
- **Virtual env**: Luôn activate `source ~/An_solo/detector/venv/bin/activate` trước khi chạy.
- **File PE lỗi**: Tất cả script đều skip file không parse được (không crash).
- **Seed cố định**: Mọi script dùng `--seed 42` để đảm bảo reproducibility.
- **Family-disjoint**: Dataset đã được chia train/test thủ công. Script không tự split để tránh rò rỉ họ malware giữa train và test.
