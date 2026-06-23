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

Chạy nền bằng `nohup` (test chỉ chạy nếu train thành công):

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_deepmd.py &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_deepmd.py
' > deepmd.log 2>&1 &
echo $!
```

Theo dõi log:

```bash
tail -f deepmd.log
```

Mặc định train `MalConvPlus`. Có thể truyền cùng hyperparameter model cho cả train và test khi chạy biến thể khác.

## CNN Keras gốc

Adapter chuyển PE sang ảnh bằng thuật toán trong `utils/data_conversion.ipynb`. Keras resize ảnh thành RGB 256×256 và model giữ nguyên kiến trúc, loss, optimizer, class weight, batch size và 10 epoch của notebook combined classifier.

```bash
python reproduction/train_cnn.py
python reproduction/test_cnn.py
```

Chạy nền bằng `nohup`:

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_cnn.py &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_cnn.py
' > cnn.log 2>&1 &
echo $!

tail -f cnn.log
```

## EMBER2024 example gốc

Adapter tạo `X_train.dat/y_train.dat` và `X_test.dat/y_test.dat`, sau đó train bằng `ember2024/examples/train_lgbm.py` và config gốc.

```bash
python reproduction/train_ember2024.py --task binary
python reproduction/test_ember2024.py --task binary

python reproduction/train_ember2024.py --task family
python reproduction/test_ember2024.py --task family
```

Chạy nền binary bằng `nohup`:

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_ember2024.py --task binary &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_ember2024.py --task binary
' > ember_binary.log 2>&1 &
echo $!
```

Chạy nền family bằng `nohup`:

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_ember2024.py --task family &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_ember2024.py --task family
' > ember_family.log 2>&1 &
echo $!
```

Theo dõi log tương ứng:

```bash
tail -f ember_binary.log
tail -f ember_family.log
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

Chạy nền bằng `nohup`:

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_sorel.py &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_sorel.py
' > sorel.log 2>&1 &
echo $!

tail -f sorel.log
```

Đây là tái huấn luyện SOREL `PENetwork` trên dataset riêng, không phải tái tạo đầy đủ thí nghiệm SOREL-20M/ALOHA.

## SOREL-20M family adaptation (chạy riêng)

Pipeline này giữ backbone `PENetwork` của SOREL nhưng thay malware binary head bằng một linear head dự đoán `Benign` và các malware family. Không thêm pipeline này vào `train_all.sh`/`test_all.sh`; chạy riêng để không làm luồng reproduction tổng dài hơn.

```bash
python reproduction/train_sorel_family.py
python reproduction/test_sorel_family.py
```

Chạy nền bằng `nohup`:

```bash
nohup bash -c '
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/train_sorel_family.py &&
  PYTHONUNBUFFERED=1 venv/bin/python reproduction/test_sorel_family.py
' > sorel_family.log 2>&1 &
echo $!

tail -f sorel_family.log
```

Output mặc định nằm trong `reproduction_output/sorel_family/`. Đây là family adaptation, không phải training objective gốc của SOREL-20M.

Nếu lỗi `Duplicate SHA-256 samples found while preparing SOREL family data`, nghĩa là cùng một PE xuất hiện nhiều lần ở các split hoặc family khác nhau sau khi tính SHA-256. Script sẽ ghi danh sách trùng cần xử lý vào:

```text
reproduction_output/sorel_family/data/duplicate_sha256.json
```

Không nên bỏ qua lỗi này nếu trùng giữa train và test, vì nó làm rò rỉ test. Cần mở file JSON trên, xem `first` và `duplicate`, rồi xóa hoặc di chuyển mẫu trùng trong dataset thật trước khi chạy lại.

Nếu duplicate nằm trong cùng split, cùng family và cùng nhãn, script tự bỏ qua bản lặp và ghi vào:

```text
reproduction_output/sorel_family/data/skipped_duplicate_sha256.json
```

## Chạy tuần tự tất cả

```bash
chmod +x reproduction/train_all.sh reproduction/test_all.sh
bash reproduction/train_all.sh 2>&1 | tee reproduction_train.log
bash reproduction/test_all.sh 2>&1 | tee reproduction_test.log
```

Chạy toàn bộ train → test dưới nền:

```bash
nohup bash -c '
  bash reproduction/train_all.sh &&
  bash reproduction/test_all.sh
' > reproduction_all.log 2>&1 &
echo $!

tail -f reproduction_all.log
```

Các lệnh trên cần được chạy từ thư mục gốc `~/An_solo/detector`. Dùng `&&` để không chạy test khi train lỗi. `echo $!` in PID của tiến trình nền; có thể dừng bằng `kill <PID>`.

Mỗi test ghi `test_results.json` trong thư mục output tương ứng. Không có kết quả nào được điền sẵn; file kết quả chỉ xuất hiện sau khi model chạy thật.

## Xem toàn bộ metrics

Sau khi một hoặc nhiều model đã test xong, chạy:

```bash
python reproduction/report_metrics.py
```

Script tự đọc kết quả của DeepMD, CNN, EMBER2024 binary/family, SOREL binary và SOREL family adaptation. Với mỗi model đã có kết quả, script in:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Precision/Recall/F1 và support từng class
- Confusion Matrix

Model chưa có `test_results.json` sẽ được báo `[SKIP]` và không ngăn kết quả của các model khác hiển thị. Nếu output nằm ở vị trí khác mặc định:

```bash
python reproduction/report_metrics.py --output-root /duong/dan/reproduction_output
```

## Mức độ sử dụng code gốc và các thay đổi

Phần này mô tả chính xác mức độ tái sử dụng source của từng project. Khi viết báo cáo, không nên gọi chung cả bốn pipeline là "chạy nguyên bản 100%" vì mức độ can thiệp không giống nhau.

### Bảng tóm tắt

| Project | Phần huấn luyện | Phần được giữ nguyên | Phần được viết/chỉnh lại | Cách nên mô tả trong báo cáo |
|---|---|---|---|---|
| deep-malware-detection | Gọi trực tiếp trainer gốc | Model, loss, optimizer, early stopping, loader và trích PE header | Chuẩn bị thư mục input và evaluator cho test ngoài | Chạy source gốc trên dataset riêng với adapter dữ liệu |
| malware-classification-CNN | Chuyển notebook thành script local | Kiến trúc Keras, preprocessing ảnh đầu vào, loss, optimizer, class weight và số epoch | Đường dẫn, khám phá class, generator theo thư mục, lưu model và test | Tái triển khai trung thực notebook gốc thành script local |
| ember2024 | Gọi trực tiếp example gốc | `train_lgbm.py`, `thrember.train_model()` và config LightGBM gốc | Adapter tạo memmap và evaluator không cần challenge set | Chạy example gốc trên feature EMBER v3 trích từ dataset riêng |
| SOREL-20M | Gọi hàm train và `PENetwork` gốc | Backbone, malware head, optimizer, malware loss và LMDB reader | Feature v3, schema metadata tối giản, bỏ auxiliary targets và evaluator mới | Tái huấn luyện SOREL PENetwork binary trên dataset riêng; không phải full SOREL reproduction |
| SOREL-20M family | Script family riêng | `PENetwork.model_base`, LMDB feature reader và EMBER v3 feature | `family_label`, linear family head, weighted `CrossEntropyLoss`, evaluator multiclass | SOREL PENetwork family adaptation trên dataset riêng |

### 1. deep-malware-detection

#### Code gốc được chạy trực tiếp

`reproduction/train_deepmd.py` gọi trực tiếp các file sau mà không chép lại kiến trúc:

- `deep-malware-detection/src/bin/extract_header.py`
- `deep-malware-detection/src/deep_malware_detection/train.py`
- `deep-malware-detection/src/deep_malware_detection/models.py`
- `deep-malware-detection/src/deep_malware_detection/dataset.py`
- `deep-malware-detection/src/deep_malware_detection/utils.py`

Do đó các thành phần sau giữ nguyên theo repository gốc:

- `MalConvBase` và `MalConvPlus`;
- cách padding PE header tới 4.096 byte;
- `BCEWithLogitsLoss`;
- Adam optimizer;
- `ReduceLROnPlateau`;
- early stopping và cách lưu checkpoint;
- cách code gốc tạo validation/test nội bộ từ tập train đầu vào.

#### Phần adapter được thêm

- Đọc đúng bốn thư mục train/test của dataset người dùng.
- Gộp các malware family vào nhãn binary `malware` mà không sửa file nguồn.
- Đổi tên bản sao bằng SHA-256 để các family có file trùng tên không ghi đè nhau.
- Gọi `extract_header.py` gốc để tạo pickle.
- Trích xuất cả external test split để đánh giá cuối cùng.
- `reproduction/test_deepmd.py` là evaluator mới vì `train.py` gốc không có CLI nhận một external test directory riêng.

Evaluator mới chỉ load checkpoint và tính accuracy, F1, ROC AUC, confusion matrix; nó không cập nhật trọng số.

#### Khác biệt so với thí nghiệm công bố của repository

- Dùng dataset riêng thay vì dataset tác giả đã sử dụng.
- Thành phần family bị gộp thành bài toán benign/malware đúng với output binary của model.
- Tỷ lệ và phân bố class phụ thuộc dataset riêng.

Cách mô tả đề xuất:

> Chúng tôi chạy trực tiếp implementation PyTorch gốc của deep-malware-detection. Một adapter chỉ được sử dụng để ánh xạ cây thư mục dataset riêng sang hai lớp benign/malware và gọi bộ trích PE header đi kèm repository. Kiến trúc và training loop không bị thay đổi.

### 2. malware-classification-CNN

#### Nguồn gốc script

Repository gốc không cung cấp script Python train local; code nằm trong:

- `malware-classification-CNN/combined_classifier/combined_classifier_test.ipynb`
- `malware-classification-CNN/utils/data_conversion.ipynb`

`reproduction/train_cnn.py` là bản chuyển notebook Keras thành script local. Nó không gọi notebook trực tiếp, nhưng giữ lại:

- input RGB 256 x 256;
- rescale pixel bằng `1/255`;
- bốn tầng convolution lần lượt có 64, 32, 32 và 16 filter;
- kernel 3 x 3, ReLU và MaxPooling 2 x 2;
- Dropout 0,25 trước Flatten;
- Dense 128 + Dropout 0,25;
- Dense 50 + Dropout 0,5;
- output Softmax theo số family thực tế;
- categorical cross-entropy;
- Adam mặc định của Keras;
- balanced class weights;
- batch size 32 và mặc định 10 epoch.

#### Phần thay đổi để chạy local

- Bỏ mount Google Drive và thay toàn bộ đường dẫn Colab bằng CLI arguments.
- Dò class từ tên thư mục thay vì dictionary 26 class Malimg được hard-code trong notebook.
- Dùng `flow_from_directory()` thay cho `flow_from_dataframe()` vì dataset người dùng đã tổ chức theo thư mục class và không có các CSV Malimg gốc.
- Không gộp `val_df` vào train vì dataset người dùng chỉ có train/test đã chia sẵn.
- Lưu final model dưới định dạng `.keras`, class mapping và history dưới JSON.
- Notebook test gốc tạo `ModelCheckpoint` theo `val_accuracy` dù không truyền validation data. Script local không giữ callback không hoạt động này và lưu final model sau epoch cuối.
- `reproduction/test_cnn.py` là evaluator mới, dùng external test split và không cập nhật trọng số.

#### Chuyển PE thành ảnh

`reproduction/cnn_data.py` thực hiện trực tiếp thuật toán trong notebook:

1. đọc PE thành dãy byte;
2. chỉ giữ các hàng đủ 16 byte;
3. chọn chiều rộng là lũy thừa của 2 dựa trên căn bậc hai số byte;
4. reshape thành ảnh xám và lưu PNG;
5. Keras đọc PNG thành RGB rồi resize 256 x 256 như generator notebook.

Bản script bỏ file `.bytes` trung gian nhưng tạo cùng ma trận byte đầu vào; thay đổi này chỉ loại bỏ I/O trung gian, không thêm feature mới.

Cách mô tả đề xuất:

> Notebook TensorFlow/Keras gốc được chuyển thành script local. Kiến trúc, loss, optimizer, class weighting và preprocessing model được bảo toàn. Các thay đổi chỉ liên quan đến đường dẫn, phát hiện class động, bỏ phụ thuộc CSV/Google Drive và lưu output theo định dạng local.

### 3. ember2024

#### Code gốc được chạy trực tiếp

Sau khi adapter tạo dữ liệu, `reproduction/train_ember2024.py` chạy trực tiếp:

- `ember2024/examples/train_lgbm.py`
- `thrember.train_model()` trong `ember2024/src/thrember/model.py`
- `ember2024/examples/lgbm_config.json` cho binary;
- `ember2024/examples/lgbm_config_family.json` cho family.

Các hyperparameter LightGBM trong config không được thay bằng config của wrapper cũ. Hàm gốc vẫn lấy 10% của train làm validation nội bộ bằng stratified split.

#### Phần adapter được thêm

- `reproduction/ember_data.py` đọc PE trong train/test đã chia sẵn.
- Gọi trực tiếp `PEFeatureExtractor.feature_vector()` của `thrember` để trích EMBER v3.
- Ghi đúng format memmap mà `thrember.read_vectorized_features()` yêu cầu:
  - `X_train.dat`, `y_train.dat`;
  - `X_test.dat`, `y_test.dat`.
- File PE không trích được feature bị bỏ cùng nhãn tương ứng và được ghi vào `failed_train.json` hoặc `failed_test.json`.
- Với family task, adapter tạo một class mapping thống nhất cho train/test.

Dataset EMBER2024 công bố ban đầu dùng raw JSONL chứa metadata và raw features. Dataset người dùng chỉ có PE và nhãn thư mục, nên adapter tạo vector trực tiếp thay vì tạo metadata JSONL giả. Feature vector cuối cùng vẫn do implementation `thrember` gốc sinh ra.

`ember2024/examples/eval_lgbm.py` gốc yêu cầu thêm challenge set, trong khi dataset người dùng không có challenge split. Vì vậy `reproduction/test_ember2024.py` là evaluator mới chỉ đánh giá external test split thật; không tạo challenge data giả.

Cách mô tả đề xuất:

> Chúng tôi sử dụng trực tiếp PEFeatureExtractor, hàm train_model, example huấn luyện và cấu hình LightGBM của EMBER2024. Một adapter được bổ sung để vector hóa các PE trong dataset riêng thành memmap đầu vào chính thức của thrember. Script đánh giá được giới hạn ở test split do dataset không có challenge set.

### 4. SOREL-20M

#### Code gốc được tái sử dụng

Pipeline gọi trực tiếp:

- `PENetwork` trong `SOREL-20M/nets.py`;
- `train_network()` và `compute_loss()` trong `SOREL-20M/train.py`;
- `Dataset` và phép hậu xử lý feature trong `SOREL-20M/dataset.py`;
- `GeneratorFactory` trong `SOREL-20M/generators.py`;
- Adam optimizer và malware binary cross-entropy của code gốc.

#### Format được xây theo yêu cầu SOREL

`reproduction/sorel_data.py` tạo:

- LMDB `ember_features`, dùng SHA-256 làm key;
- value được đóng gói bằng msgpack và zlib theo format mà `LMDBReader` gốc đọc;
- SQLite `meta.db` với bảng `meta`;
- các cột trung thực có thể lấy từ dataset: `sha256`, `is_malware`, `rl_fs_t`.

Timestamp `rl_fs_t` không phải thời điểm malware xuất hiện thật. Nó chỉ là giá trị kỹ thuật để ánh xạ thư mục `train` và `test` có sẵn vào điều kiện truy vấn split của SOREL. Không được dùng timestamp này cho phân tích thời gian.

#### Những điểm không thể giữ như SOREL-20M đầy đủ

- SOREL gốc dùng EMBER v2 gồm 2.381 feature; repository hiện tại không chứa extractor EMBER v2 tương thích dataset riêng.
- Pipeline dùng EMBER v3 do `thrember` sinh ra và truyền dimension thực tế, hiện là 2.386, vào `PENetwork`.
- Dataset không có `rl_ls_const_positives`, do đó không huấn luyện count loss.
- Dataset không có 11 SMART tags, do đó không huấn luyện tag loss.
- Không tạo nhãn count/tag bằng số 0 vì việc đó sẽ biến dữ liệu thiếu nhãn thành nhãn âm giả.
- `PENetwork` trong source gốc vẫn khởi tạo count/tag heads; do hai label tương ứng không được trả về, hai head này không tham gia loss và không được dùng khi báo cáo.
- Dataset không có validation split riêng. SQLite chỉ ánh xạ train/test đã có; validation loader của SOREL không có mẫu.
- `reproduction/test_sorel.py` là evaluator binary mới vì evaluator gốc hard-code feature dimension 2.381 và mặc định yêu cầu cả count/tags.

Do thay EMBER v2 bằng EMBER v3 và bỏ hai auxiliary task, pipeline này không phải reproduction đầy đủ benchmark SOREL-20M. Nó chỉ kiểm nghiệm kiến trúc `PENetwork` và malware head gốc trên dataset riêng.

Cách mô tả đề xuất:

> Chúng tôi tái sử dụng backbone PENetwork, malware head, binary loss và training function của SOREL-20M. Dữ liệu riêng được đóng gói theo interface LMDB/SQLite của repository. Do thiếu EMBER v2, vendor detection counts và SMART tags, thí nghiệm sử dụng EMBER v3 và chỉ tối ưu malware binary objective; vì vậy kết quả được báo cáo là SOREL PENetwork binary adaptation, không phải full SOREL-20M reproduction.

#### Family adaptation chạy riêng

`reproduction/train_sorel_family.py` và `reproduction/test_sorel_family.py` thêm một bài toán family riêng cho SOREL:

- `reproduction/sorel_family_data.py` tạo LMDB/SQLite giống interface SOREL nhưng thêm `family_label` và `family_names.json`;
- class `Benign` được đưa vào mapping cùng các malware family;
- `reproduction/sorel_family_model.py` dùng `PENetwork.model_base` làm backbone và thêm linear head cho số family thực tế;
- loss là weighted `CrossEntropyLoss`, không dùng malware/count/tag losses của SOREL gốc;
- evaluator ghi accuracy, macro precision/recall/F1, classification report và confusion matrix multiclass.

#### Những thứ đã dùng để chuyển SOREL sang family

Phần family adaptation dùng lại các thành phần sau từ SOREL và các adapter hiện có:

- `PENetwork.model_base`: giữ phần backbone FFNN của SOREL để mã hóa vector feature;
- `LMDBReader` và `features_postproc_func` trong `SOREL-20M/dataset.py`: giữ cách đọc và hậu xử lý feature giống SOREL;
- EMBER v3 feature từ `thrember`: dùng cùng nguồn feature đã dùng cho SOREL binary adaptation;
- format LMDB `ember_features/`: vẫn lưu feature bằng SHA-256, msgpack và zlib để tương thích reader gốc;
- SQLite `meta.db`: vẫn dùng `sha256`, `is_malware`, `rl_fs_t` để bám interface split của SOREL;
- tên thư mục malware family trong dataset thật: dùng làm nhãn family trung thực;
- class `Benign`: đưa vào family mapping như một class riêng, đứng đầu `family_names.json`;
- weighted `CrossEntropyLoss`: thay binary malware loss bằng multiclass loss có cân bằng class;
- `classification_report` và `confusion_matrix` của scikit-learn: dùng để báo cáo metric multiclass.

Các phần đã thay hoặc không dùng:

- không dùng malware binary head làm output cuối;
- không dùng count head, tag head, count loss hoặc tag loss vì dataset không có nhãn thật tương ứng;
- không dùng `train_network()` gốc vì hàm này chỉ tối ưu các head gốc của SOREL;
- thêm `family_label` vào SQLite để dataset trả về nhãn family;
- thêm linear family head mới trên backbone để sinh logits theo số family thực tế.

Cách mô tả đề xuất:

> Chúng tôi xây dựng một family-classification adaptation riêng từ SOREL PENetwork bằng cách giữ backbone feature của PENetwork và thay output bằng linear head đa lớp cho `Benign` và các malware family. Đây là thí nghiệm mở rộng trên dataset riêng, không phải objective gốc của SOREL-20M.

### Nguyên tắc báo cáo kết quả

- Chỉ ghi số liệu sau khi `test_*.py` chạy thật và sinh `test_results.json`.
- Báo rõ dataset, số mẫu train/test thực tế và số PE bị feature extractor bỏ qua.
- Không so sánh trực tiếp số accuracy/AUC với paper như thể cùng dataset.
- Với DeepMD, ghi là original implementation + data adapter.
- Với CNN, ghi là faithful notebook-to-script conversion.
- Với EMBER2024, ghi là original feature extractor/trainer/config + data adapter.
- Với SOREL, luôn ghi rõ binary adaptation, EMBER v3 và không có auxiliary losses.
- Với SOREL family, ghi rõ đây là family adaptation riêng, không phải training objective gốc của SOREL-20M.
