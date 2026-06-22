# Automated reproduction pipelines

Các script trong thư mục này chỉ đọc dataset đã chia sẵn của người dùng:

```text
~/An_solo/dataset/
├── Virus/
│   ├── Virus train/<family>/*.{exe,dll}
│   └── Virus test/<family>/*.{exe,dll}
└── Benign/
    ├── Benign train/*.{exe,dll}
    └── Benign test/*.{exe,dll}
```

Không gộp train/test và không dùng test để cập nhật trọng số. DeepMD và EMBER2024 vẫn thực hiện validation nội bộ theo training loop gốc của chúng.

## Cài môi trường Linux

Khuyến nghị Python 3.10 hoặc 3.11 vì TensorFlow và SOREL bản cũ không phù hợp Python 3.13.

```bash
cd ~/An_solo/detector
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch tensorflow scikit-learn pandas pillow tqdm pefile \
  lightgbm lmdb msgpack logzero baker matplotlib seaborn
python -m pip install -e ./ember2024
```

Có thể đổi đường dẫn bằng biến môi trường:

```bash
export SD_FSPROTO_DATASET="$HOME/An_solo/dataset"
export SD_FSPROTO_OUTPUT="$HOME/An_solo/detector/reproduction_output"
```

## Deep malware detection

Train gọi trực tiếp `extract_header.py`, `train.py`, `models.py` và training loop gốc:

```bash
python reproduction/train_deepmd.py
python reproduction/test_deepmd.py
```

Mặc định train `MalConvPlus`. Có thể truyền cùng hyperparameter model cho cả train và test khi chạy biến thể khác.

## CNN Keras gốc

Adapter chuyển PE sang ảnh bằng thuật toán trong `utils/data_conversion.ipynb`. Keras resize ảnh thành RGB 256×256 và model giữ nguyên kiến trúc, loss, optimizer, class weight, batch size và 10 epoch của notebook combined classifier.

```bash
python reproduction/train_cnn.py
python reproduction/test_cnn.py
```

## EMBER2024 example gốc

Adapter tạo `X_train.dat/y_train.dat` và `X_test.dat/y_test.dat`, sau đó train bằng `ember2024/examples/train_lgbm.py` và config gốc.

```bash
python reproduction/train_ember2024.py --task binary
python reproduction/test_ember2024.py --task binary

python reproduction/train_ember2024.py --task family
python reproduction/test_ember2024.py --task family
```

## SOREL-20M

Adapter tạo `ember_features/` LMDB và `meta.db` SQLite. Do dataset không có EMBER v2, detection count và SMART tags:

- dùng feature EMBER v3 thật từ `thrember`;
- truyền đúng feature dimension vào `PENetwork` gốc;
- chỉ malware binary label tham gia loss;
- không tạo nhãn count/tag giả.

```bash
python reproduction/train_sorel.py
python reproduction/test_sorel.py
```

Đây là tái huấn luyện SOREL `PENetwork` trên dataset riêng, không phải tái tạo đầy đủ thí nghiệm SOREL-20M/ALOHA.

## Chạy tuần tự tất cả

```bash
chmod +x reproduction/train_all.sh reproduction/test_all.sh
bash reproduction/train_all.sh 2>&1 | tee reproduction_train.log
bash reproduction/test_all.sh 2>&1 | tee reproduction_test.log
```

Mỗi test ghi `test_results.json` trong thư mục output tương ứng. Không có kết quả nào được điền sẵn; file kết quả chỉ xuất hiện sau khi model chạy thật.
