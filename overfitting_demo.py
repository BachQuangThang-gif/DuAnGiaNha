import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
data = pd.read_csv('house_prices.csv')

X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Chia tập Train (70%) và Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 60)
print("BÀI THÍ NGHIỆM: OVERFITTING VÀ KỸ THUẬT REGULARIZATION")
print("=" * 60)

# ==========================================
# GIAI ĐOẠN 1: TẠO HIỆN TƯỢNG OVERFITTING (QUÁ KHỚP)
# Dùng Đa thức bậc 4 (Polynomial degree = 4) trên tập dữ liệu nhỏ
# ==========================================
poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

overfit_model = LinearRegression()
overfit_model.fit(X_train_poly, y_train)

# Đánh giá mô hình Overfitting
train_pred_overfit = overfit_model.predict(X_train_poly)
test_pred_overfit = overfit_model.predict(X_test_poly)

print("\n--- 1. KẾT QUẢ MÔ HÌNH OVERFITTING (BẬC 4) ---")
print(f"Số lượng đặc trưng tạo ra: {X_train_poly.shape[1]}")
print(f"[Train] R2 Score: {r2_score(y_train, train_pred_overfit):.4f}  | MSE: {mean_squared_error(y_train, train_pred_overfit):.2f}")
print(f"[Test]  R2 Score: {r2_score(y_test, test_pred_overfit):.4f} | MSE: {mean_squared_error(y_test, test_pred_overfit):.2f}")
print("-> Nhận xét: Điểm Train gần như hoàn hảo (1.0), nhưng điểm Test âm rất sâu hoặc MSE cực lớn.")
print("   Đây chính là biểu hiện rõ ràng nhất của việc học vẹt (Overfitting)!")

# ==========================================
# GIAI ĐOẠN 2: KHẮC PHỤC OVERFITTING
# Giải pháp: Chuẩn hóa đặc trưng (StandardScaler) + Hồi quy Ridge (L2 Penalty)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Ridge thêm hệ số phạt alpha để kìm hãm trọng số không bị bùng nổ
ridge_model = Ridge(alpha=100.0)
ridge_model.fit(X_train_scaled, y_train)

train_pred_ridge = ridge_model.predict(X_train_scaled)
test_pred_ridge = ridge_model.predict(X_test_scaled)

print("\n--- 2. KẾT QUẢ SAU KHI DÙNG RIDGE REGULARIZATION ---")
print(f"[Train] R2 Score: {r2_score(y_train, train_pred_ridge):.4f}  | MSE: {mean_squared_error(y_train, train_pred_ridge):.2f}")
print(f"[Test]  R2 Score: {r2_score(y_test, test_pred_ridge):.4f}  | MSE: {mean_squared_error(y_test, test_pred_ridge):.2f}")
print("-> Kết quả: Sai số trên tập Test đã giảm mạnh, R2 tập Test hồi phục về mức dương cao.")
print("   Mô hình đã học cách khái quát hóa thay vì học vẹt.")
print("=" * 60)