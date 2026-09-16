# Dự Án Dự Báo Giá Nhà (House Price Prediction)

Dự án Machine Learning sử dụng thuật toán Hồi quy tuyến tính (Linear Regression) để dự đoán giá nhà.

## Thành phần dự án

- `house_prices.csv`: Bộ dữ liệu diện tích, số phòng và giá.
- `main.py`: Mã nguồn huấn luyện mô hình.

## Cách chạy chương trình

1. Cài thư viện:
   pip install pandas scikit-learn
2. Chạy file:
   python main.py

## Thí nghiệm Overfitting và Kỹ thuật Tránh Overfitting

- File thực nghiệm: `overfitting_demo.py`
- **Tạo Overfitting**: Sử dụng hồi quy đa thức bậc cao (Polynomial Features bậc 4) khiến mô hình cố khớp toàn bộ dữ liệu huấn luyện (Train R2 = 1.0 nhưng Test R2 bị âm nặng).
- **Khắc phục**: Sử dụng chuẩn hóa `StandardScaler` kết hợp kỹ thuật điều chuẩn **Ridge Regression (L2 Regularization)** với \(\alpha = 100\) để phạt các trọng số quá lớn, giúp mô hình tổng quát hóa tốt trên tập kiểm thử (Test).
