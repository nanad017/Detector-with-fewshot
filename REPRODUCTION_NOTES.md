# Ghi chú tái huấn luyện các mô hình gốc

## Mục tiêu

Tái huấn luyện bốn project trên dataset riêng, ưu tiên sử dụng kiến trúc, preprocessing, feature extraction và training loop gốc:

1. `deep-malware-detection`
2. `malware-classification-CNN`
3. `ember2024`
4. `SOREL-20M`

Không được công bố hoặc giả lập kết quả khi chưa chạy huấn luyện và đánh giá thật trên máy Linux chứa dataset.

## Môi trường và dataset

- Dataset thật nằm trên máy Linux, không nằm trong workspace Windows hiện tại.
- Cấu trúc đang được sử dụng:
  - `~/An_solo/dataset/Virus/Virus train/<family>/`
  - `~/An_solo/dataset/Virus/Virus test/<family>/`
  - `~/An_solo/dataset/Benign/Benign train/`
  - `~/An_solo/dataset/Benign/Benign test/`
- Dataset đã được chia train/test sẵn; không gộp rồi chia lại.
- Test phải được giữ riêng để đánh giá cuối cùng, không dùng để cập nhật trọng số hoặc lựa chọn mô hình.

## Validation và test có làm thay đổi trọng số không?

- **Train**: có cập nhật trọng số bằng backpropagation và optimizer.
- **Validation**: không trực tiếp cập nhật trọng số. Tuy nhiên, kết quả validation có thể được dùng để chọn checkpoint, early stopping hoặc điều chỉnh learning rate.
- **Test**: không cập nhật trọng số và chỉ được dùng để báo cáo kết quả cuối cùng.

Một số project gốc bắt buộc lấy một phần train làm validation nội bộ. Việc này không thay đổi test đã chia sẵn, nhưng làm giảm số mẫu trực tiếp dùng để cập nhật trọng số. Cần xác nhận có cho phép validation nội bộ hay không trước khi hoàn thiện pipeline.

## 1. deep-malware-detection

### File gốc cần dùng

- `deep-malware-detection/src/bin/extract_header.py`
- `deep-malware-detection/src/deep_malware_detection/train.py`
- `deep-malware-detection/src/deep_malware_detection/models.py`
- `deep-malware-detection/src/deep_malware_detection/dataset.py`
- `deep-malware-detection/src/deep_malware_detection/utils.py`

### Kết luận

- Có thể chạy trực tiếp code và model gốc.
- Input gốc là PE header được chuyển thành pickle bằng `extract_header.py`.
- `train.py` gốc tự chia dữ liệu đầu vào thành train/validation/test.
- Truyền `--test_size 0.0` vào code gốc gây lỗi vì `sklearn.train_test_split` không chấp nhận `test_size=0.0`.
- Bản sửa tạm thời cho `dataset.py` đã được gỡ; file hiện khớp lại với phiên bản trong repository.
- Muốn giữ code hoàn toàn nguyên bản phải dùng tỷ lệ split hợp lệ mà code gốc hỗ trợ.
- Đã thêm DeepMD family adaptation chạy riêng dưới `reproduction/`, giữ `extract_header.py` và MalConv backbone nhưng thay binary head/loss bằng multiclass family head và weighted `CrossEntropyLoss`.
- Pipeline family này không thay thế pipeline DeepMD binary gốc; khi báo cáo phải gọi rõ là adaptation riêng trên dataset riêng.

## 2. malware-classification-CNN

### File gốc cần dùng

- `malware-classification-CNN/combined_classifier/combined_classifier_test.ipynb`
- `malware-classification-CNN/combined_classifier/combined_classifier_val.ipynb`
- `malware-classification-CNN/utils/data_conversion.ipynb`

### Kiến trúc gốc

- Framework: TensorFlow/Keras, không phải PyTorch.
- Input: ảnh RGB `256 x 256`, rescale `1/255`.
- Các lớp:
  - `Conv2D(64, 3x3, relu)` + MaxPool
  - `Conv2D(32, 3x3, relu)` + MaxPool
  - `Conv2D(32, 3x3, relu)` + MaxPool
  - `Conv2D(16, 3x3, relu)` + MaxPool
  - Dropout `0.25`
  - Flatten
  - Dense `128`, relu + Dropout `0.25`
  - Dense `50`, relu + Dropout `0.5`
  - Dense `num_classes`, softmax
- Loss: categorical cross-entropy.
- Optimizer: Adam mặc định của Keras.
- Batch size: 32.
- Epoch: 10.
- Dùng balanced class weights.

### Vấn đề hiện tại

- `scripts/train_cnn.py` là bản viết lại bằng PyTorch, không phải code gốc.
- Khác framework, input grayscale thay vì RGB và training loop khác; không được dùng kết quả này để tuyên bố là mô hình CNN gốc.
- Dataset chỉ có train/test nên luồng phù hợp là notebook `combined_classifier_test.ipynb`, không tự tạo validation nếu chưa được cho phép.
- Notebook gốc được viết cho Google Colab và có đường dẫn Google Drive; cần chuyển thành script local nhưng giữ nguyên model và logic huấn luyện.
- Chuyển PE thành ảnh phải bám thuật toán trong `data_conversion.ipynb`, không tự thay bằng preprocessing khác mà không giải thích.

## 3. ember2024

### File gốc cần dùng

- `ember2024/examples/train_lgbm.py`
- `ember2024/examples/lgbm_config.json`
- `ember2024/examples/lgbm_config_family.json`
- `ember2024/src/thrember/features.py`
- `ember2024/src/thrember/model.py`

### Kết luận

- `scripts/train_ember_lgbm.py` là wrapper viết mới với tham số LightGBM khác, không phải example gốc.
- Example gốc gọi `thrember.train_model()` và đọc:
  - `X_train.dat`
  - `y_train.dat`
- Cần adapter tạo đúng memmap `.dat` từ dataset riêng rồi gọi nguyên `examples/train_lgbm.py` cùng config gốc.
- `thrember.train_model()` tự lấy 10% train làm validation bằng stratified split.
- Nếu cấm mọi validation nội bộ thì phải sửa logic gốc; khi đó không còn chạy nguyên bản hoàn toàn.

## 4. SOREL-20M

### File gốc cần dùng

- `SOREL-20M/train.py`
- `SOREL-20M/nets.py`
- `SOREL-20M/dataset.py`
- `SOREL-20M/generators.py`
- `SOREL-20M/config.py`

### Format dữ liệu gốc

- LMDB `ember_features`: lưu vector feature, key là SHA-256.
- SQLite `meta.db`: bảng `meta` chứa SHA-256, nhãn malware, timestamp split, detection count và 11 SMART tags.
- Feature gốc là EMBER v2, 2.381 chiều.

### Giới hạn của dataset riêng

- Dataset hiện có nhãn benign/malware và malware family.
- Không có detection count thật.
- Không có 11 SMART tags thật.
- Không được tự bịa các nhãn còn thiếu.

### Hướng tái huấn luyện đã thống nhất

- Xây LMDB và SQLite từ dữ liệu thật của người dùng.
- Chỉ huấn luyện malware binary target nếu không có count/tags.
- Giữ nguyên `PENetwork` backbone và malware head gốc.
- Tắt count/tag target phải được ghi rõ; đây là tái huấn luyện kiến trúc SOREL trên dataset riêng, không phải tái tạo đầy đủ thí nghiệm SOREL-20M/ALOHA.
- `train.py` gốc hiện khởi tạo cả ba head bất kể cờ target; cần adapter hoặc thay đổi tối thiểu để các cờ thực sự điều khiển head. Mọi thay đổi phải được ghi lại.
- SOREL dùng timestamp trong SQLite để xác định train/validation/test. Cần ánh xạ split có sẵn của dataset mà không làm rò rỉ test.

## Các lỗi đã gặp trong quá trình kiểm kê

### Thiếu `gh`

- Lệnh: `gh issue list --repo nanad017/Detector-with-fewshot --limit 30 --state all`
- Lỗi: `gh` không được nhận diện.
- Nguyên nhân dự đoán: GitHub CLI chưa được cài trong môi trường Windows hiện tại.
- Cách xử lý: chưa đọc được issue tracker; không giả định nội dung issue.

### Thiếu `jq`

- Lệnh: `jq --version`
- Lỗi: `jq` không được nhận diện.
- Nguyên nhân dự đoán: `jq` chưa được cài.
- Cách xử lý: dùng PowerShell `ConvertFrom-Json` để đọc notebook, không sửa notebook gốc.

### Wildcard kiểu Unix với `rg` trên Windows

- Lệnh có dạng: `rg ... malware-classification-CNN/*/*.ipynb`
- Lỗi: tên file/thư mục không hợp lệ trên Windows.
- Nguyên nhân dự đoán: PowerShell/Windows không mở rộng glob theo cách shell Unix mong đợi.
- Cách xử lý: dùng `Get-ChildItem` để liệt kê notebook rồi đọc từng file.

### DeepMD với `test_size=0.0`

- Lệnh được cấu hình trong `scripts/train_all.sh`: `python train.py ... --val_size 0.15 --test_size 0.0`
- Lỗi dự kiến từ code đã kiểm tra: `train_test_split` không chấp nhận `test_size=0.0`.
- Nguyên nhân: code gốc luôn gọi `train_test_split` cho test set.
- Cách sửa nếu giữ code gốc: dùng tỷ lệ test hợp lệ.
- Cách sửa thay thế: sửa loader để hỗ trợ test rỗng, nhưng phương án này không còn hoàn toàn nguyên bản.

## File phụ trợ hiện có cần xem xét

Các file sau là reimplementation/wrapper, không phải code huấn luyện gốc:

- `scripts/train_cnn.py`
- `scripts/eval_cnn.py`
- `scripts/train_sorel_ffnn.py`
- `scripts/eval_sorel_ffnn.py`
- `scripts/train_ember_lgbm.py`
- `scripts/eval_ember_lgbm.py`
- `scripts/train_all.sh`
- `scripts/test_all.sh`

Không xóa các file này khi chưa có xác nhận của người dùng. Sau khi pipeline nguyên bản hoàn thành, cần đề xuất rõ file nào không còn cần thiết và hỏi trước khi xóa.

## Trạng thái hiện tại

- Đã đọc và xác định các entry point gốc.
- Đã tạo các cặp train/test tự động trong `reproduction/` cho DeepMD binary/family, CNN Keras, EMBER2024 binary/family và SOREL binary.
- Đã tạo `reproduction/train_all.sh` và `reproduction/test_all.sh` để chạy tuần tự trên Linux.
- Chưa chạy huấn luyện thật vì dataset chỉ có trên máy Linux.
- Chưa có kết quả accuracy/AUC/F1 mới.
- Chưa được phép giả lập hoặc điền kết quả ước lượng.
- Chưa xóa file nào.
- DeepMD và EMBER2024 giữ validation nội bộ theo code gốc; test có sẵn của người dùng chỉ dùng để đánh giá cuối cùng.
