import streamlit as st

def display():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Về mặt khoa học, nghiên cứu đã bổ sung và hoàn thiện hệ thống lý luận về dự báo nhiệt độ trong bối cảnh biến đổi khí hậu tại Việt Nam, đặc biệt ở các khu vực đô thị và ven biển chịu ảnh hưởng nắng nóng cực đoan. 
            Nghiên cứu cũng xây dựng quy trình thiết kế, tiền xử lý và huấn luyện các mô hình chuỗi thời gian hiện đại như N-BEATS, TFT cũng như các mô hình học máy khác như Random Forest và XGBoost, được cải tiến phù hợp với dữ liệu khí tượng đặc thù. 
            Ngoài ra, mô hình còn sử dụng các dữ liệu đầu vào có tương quan cao với nhiệt độ như vận tốc gió, hướng gió, độ che phủ mây, lượng mưa, cùng các thuộc tính không gian (kinh độ, vĩ độ, khu vực như Hà Nội, TP.HCM) nhằm nắm bắt đặc trưng khí hậu và địa hình riêng của từng vùng. Từ đó, nghiên cứu thiết lập cơ sở đánh giá và so sánh khả năng dự báo giữa các mô hình học sâu và học máy, làm rõ ưu nhược điểm của từng phương pháp trong các điều kiện không gian, địa hình và khí hậu khác nhau. 
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Về mặt thực tiễn, nghiên cứu xây dựng bộ công cụ dự báo nhiệt độ tối đa hàng ngày phục vụ cảnh báo sớm các đợt nắng nóng tại khu vực ven biển và đô thị Việt Nam, góp phần nâng cao hiệu quả phòng chống thiên tai và bảo vệ sức khỏe cộng đồng thông qua tích hợp vào hệ thống dự báo của Tổng cục Khí tượng Thủy văn. 
            Kết quả nghiên cứu còn là cơ sở khoa học giúp các địa phương và ngành khí tượng lập kế hoạch ứng phó, giảm thiểu thiệt hại kinh tế - xã hội. Trong y tế công cộng, dự báo sớm hỗ trợ bệnh viện chủ động chuẩn bị tiếp nhận ca bệnh do nắng nóng. 
            Trong nông nghiệp, nông dân có thể lập kế hoạch tưới tiêu, thu hoạch và bảo vệ mùa màng kịp thời. Đối với ngành năng lượng, dự báo chính xác giúp điều chỉnh nhu cầu điện làm mát và phân phối điện hợp lý, giảm nguy cơ quá tải trong các đợt nắng nóng cực đoan.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {display() if name == "all" else ""}
        </div>
        ''')