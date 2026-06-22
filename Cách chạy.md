Cách chạy 
Đơn giản:


cd ~/An_solo/detector
source venv/bin/activate
nohup bash scripts/train_all.sh > train_output.log 2>&1 &
Giải thích:

nohup — không tắt khi logout
> train_output.log — ghi log ra file
2>&1 — gộp cả lỗi vào cùng file log
& — chạy ngầm
Kiểm tra sau đó:


# Xem process còn sống không
ps aux | grep train_all

# Xem log realtime
tail -f train_output.log

# Khi chạy xong, kiểm tra kết quả
ls -la models/