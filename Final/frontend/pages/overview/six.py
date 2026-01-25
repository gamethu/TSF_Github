import streamlit as st

def one():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Nghiên cứu được triển khai tại sáu khu vực của Việt Nam, bao gồm: Thanh Hóa, Quy Nhơn, Đồng Hới, Cà Mau, Tân Sơn Nhất (TP. Hồ Chí Minh) và Nội Bài (Hà Nội). 
            Các địa điểm này được lựa chọn nhằm đại diện cho những đặc trưng khí hậu và điều kiện địa lý khác nhau gồm: khu vực Bắc Trung Bộ và ven biển miền Trung (Thanh Hóa, Quy Nhơn, Đồng Hới) thường xuyên chịu ảnh hưởng của nắng nóng và bão nhiệt đới; khu vực đồng bằng và ven biển phía Nam mang đặc trưng khí hậu nhiệt đới gió mùa; trong khi đó, hai đô thị lớn là Hà Nội và TP. Hồ Chí Minh thể hiện rõ rệt hiệu ứng đảo nhiệt đô thị.
        </div>
        ''')
def two():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong khoảng thời gian từ năm 1990 đến năm 2024
        </div>
        ''')
def three():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Nghiên cứu tập trung vào việc dự báo và so sánh hiệu quả của năm mô hình học máy và học sâu hiện đại, bao gồm: Random Forest, XGBoost, Transformer, N-BEATS và Temporal Fusion Transformer (TFT) trong dự báo nhiệt độ cực đại hằng ngày (TEMP_max). Các mô hình được đánh giá theo các chỉ số: MAE, MSE, RMSE, MAPE và R².
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {one() if name == "one" else ""}
            {two() if name == "two" else ""}
            {three() if name == "three" else ""}
        </div>
        ''')