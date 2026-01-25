import streamlit as st

def display():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Phương pháp hệ thống hoá tài liệu: trọng tâm tìm kiếm sẽ tập trung vào ba hướng nghiên cứu chính: các mô hình học máy, học sâu được ứng dụng trong dự báo thời tiết; tác động của biến đổi khí hậu đối với khu vực Việt Nam; đặc thù khí hậu của các vùng ven biển cũng như đô thị thu thập các báo cáo nghiên cứu từ các cơ sở dữ liệu học thuật uy tín như Scopus, Web of Science, PubMed và Google Scholar,... 
            Đồng thời, nghiên cứu cũng sẽ tổng hợp các kiến thức lý thuyết nền tảng liên quan đến các mô hình học máy và học sâu hiện đại được ứng dụng trong chuỗi thời gian. 
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Phương pháp điều tra, khảo sát dữ liệu: đề tài nghiên cứu sẽ chủ yếu nghiên cứu sử dụng dữ liệu điều tra khảo sát của cơ quan quản lý khí quyển và đại dương quốc gia (National Oceanic and Atmospheric Administration - NOAA), Mỹ và trung tâm dự báo thời tiết tầm trung châu Âu (ECMWF) đối với các đối tượng có liên quan trực tiếp đến nhiệt độ hàng ngày tại các tỉnh thành VN và các data khí tượng khác tại các tỉnh thành Việt Nam như ERA5 – ECMWF ReAnalysis version 5 - một bộ dữ liệu khí tượng toàn cầu do ECMWF phát triển, nhằm thu thập thông tin chính xác, đa chiều và khách quan về thực trạng các đợt nắng nóng tại các khu vực ven biển và đô thị Việt Nam.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Phương pháp phân tích và xử lý dữ liệu: sau khi thu thập được số liệu khảo sát, sẽ tiến hành làm sạch và chuẩn bị dữ liệu cho mô hình học sâu thông qua các bước tiền xử lý dữ liệu, rút trích đặc trưng. 
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Phương pháp thực nghiệm và đánh giá mô hình dự đoán: cài đặt thực nghiệm, đánh giá kết quả trên tập dữ liệu thu thập.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {display() if name == "all" else ""}
        </div>
        ''')