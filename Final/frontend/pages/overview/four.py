import streamlit as st

def display():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trước hết, nhóm tiến hành thu thập, tổng hợp và xử lý dữ liệu khí tượng từ các trạm quan trắc cũng như các nguồn dữ liệu tái phân tích toàn cầu như ERA5 và MERRA-2, đồng thời kết hợp với dữ liệu viễn thám và thông tin đặc trưng khu vực nhằm xây dựng một bộ dữ liệu chuỗi thời gian đầy đủ và tin cậy cho nghiên cứu. 
            Trên cơ sở đó, nhóm sẽ sử dụng các phương pháp dự báo chuỗi thời gian, làm rõ ưu điểm và hạn chế của các mô hình Machine Learning và Deep Learning, từ đó lựa chọn các mô hình như Random Forest, XGBoost, N-BEATS, Temporal Fusion Transformer và Transformer để tiến hành khảo sát. 
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Tiếp theo, nhóm sẽ thiết kế và triển khai các dự báo nhiệt độ cực đại hằng ngày cho một số tỉnh, thành phố đại diện của cho ba miền ở Việt Nam như Nội Bài thuộc vùng Bắc Bộ nắng nóng thường xuyên, Thanh Hóa thuộc Bắc Trung Bộ nắng nóng kéo dài do công nghiệp hóa, và một số khu vực khác có vấn đề tương tự như Cà Mau, Đồng Hới, Quy Nhơn và Tân Sơn Nhất, sau đó thực hiện hiệu chỉnh tham số, huấn luyện mô hình, kiểm định và đánh giá độ chính xác theo các chỉ tiêu thống kê và chỉ số xác suất. Kết quả dự báo sẽ được phân tích, so sánh, chỉ ra những điểm mạnh, điểm yếu hoặc hạn chế của từng mô hình trong bối cảnh dữ liệu khí tượng phức tạp tại Việt Nam.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {display() if name == "all" else ""}
        </div>
        ''')