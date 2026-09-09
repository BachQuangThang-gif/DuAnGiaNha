import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Đọc dữ liệu từ file csv
data = pd.read_csv('house_prices.csv')

# 2. Tách biến đầu vào (X: Diện tích, Số ngủ, Số tắm) và biến cần dự đoán (y: Giá)
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# 3. Chia 80% dữ liệu để học (train), 20% dữ liệu để thi/kiểm tra (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Đánh giá chất lượng mô hình
y_pred = model.predict(X_test)
print("--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"R2 Score (Độ chính xác, max là 1): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (Sai số bình phương): {mean_squared_error(y_test, y_pred):.4f}")

# 6. Dự đoán thử giá một ngôi nhà mới (Diện tích 100m2, 3 phòng ngủ, 2 phòng tắm)
nha_moi = [[100, 3, 2]]
gia_du_doan = model.predict(nha_moi)
print(f"\nDự đoán giá nhà mẫu (100m2, 3 ngủ, 2 tắm): {gia_du_doan[0]:.2f} triệu VNĐ")
print(f"Hệ số góc (Weights): {model.coef_}")
print(f"Hệ số chặn (Bias): {model.intercept_}")