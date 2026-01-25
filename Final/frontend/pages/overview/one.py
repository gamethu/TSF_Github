import streamlit as st

def one():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Đầu năm 2021, một nghiên cứu của Ana Maria Vicedo-Cabrera, và cộng sự, (2021) <b style="color:blue">[1]</b> có quy mô lớn trên 732 địa điểm ở 43 quốc gia đã chỉ ra rằng 37.0% số ca tử vong liên quan đến nắng nóng trong giai đoạn 1991–2018 có thể là do biến đổi khí hậu gây ra bởi con người. 
            Bên cạnh đó, đầu năm 2020, các tác giả trong World Bank Group, (2022) <b style="color:blue">[2]</b> đã công bố Việt Nam được xếp hạng trong top 5 các quốc gia chịu tác hưởng nặng nề nhất từ biến đổi khí hậu, với tần suất và cường độ các đợt nắng nóng gia tăng đáng kể trong hai thập kỷ qua. 
            Đồng thời trong báo cáo, văn phòng Liên Hợp Quốc về giảm thiểu rủi ro thiên tai (UN International Strategy for Disaster Reduction - UNISDR) đã ước tính thiệt hại trung bình hàng năm của Việt Nam do thiên tai khoảng 2,4 tỷ USD, hoặc gần 1,5% GDP. 
            Thực tế trên đặt ra một câu hỏi lớn cho tổng cục khí tượng thủy văn và các đơn vị trực thuộc cùng hệ thống y tế công cộng về việc làm thế nào để có những biện pháp dự đoán nhiệt độ cực đại hằng ngày nhằm cảnh báo sớm và giảm thiểu rủi ro trước các đợt nắng nóng thường xuyên. 
            Để giải quyết vấn đề này, nhiều quốc gia đã tập trung vào nghiên cứu ứng dụng các mô hình học sâu vào lĩnh vực dự báo nhiệt độ. Một ví dụ tiêu biểu trong năm 2019 nghiên cứu của Bushra Shamshad, và cộng sự, (2019) <b style="color:blue">[3]</b> về một mô hình mạng nơ-ron (ANN-MLP) để dự báo các thuộc tính về thời tiết trong chuỗi thời gian 31 năm (1987-2018) đã cho thấy mô hình ANN-MLP vượt trội hơn các mô hình tuyến tính ARIMA và ETS với RMSE thấp nhất 0.73.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong bối cảnh này, Đảng và Nhà nước Việt Nam đã chủ động đưa ra những định hướng chiến lược phù hợp trong việc áp dụng công nghệ trí tuệ nhân tạo ứng phó với các đợt nắng nóng thất thường. 
            Cụ thể, trong quyết định của Thủ tướng Chính phủ, (2022) <b style="color:blue">[4]</b> phê duyệt chiến lược quốc gia về biến đổi khí hậu giai đoạn đến năm 2050 ban hành ngày 26 tháng 07 năm 2022 có mục tiêu “Trình độ khoa học và công nghệ dự báo khí tượng thủy văn, cảnh báo sớm thiên tai ngang tầm các nước phát triển khu vực châu Á; năng lực giám sát biến đổi khí hậu, quản lý rủi ro thiên tai đạt ngang tầm với các quốc gia hàng đầu trong khu vực; đáp ứng yêu cầu cung cấp dịch vụ khí hậu cơ bản.”
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Đồng thời, những tổ chức quốc tế cũng có cùng quan điểm về ứng dụng công nghệ AI trong dự báo nhiệt độ. 
            Theo kết quả báo cáo của Stephen Legg, (2021) <b style="color:blue">[5]</b> năm 2021, Ủy ban liên chính phủ về biến đổi khí hậu (Intergovernmental Panel on Climate Change - IPCC) đã khẳng định với độ tin cậy cao rằng các đợt sóng nhiệt đang ngày càng gia tăng về tần suất và cường độ, đồng thời khuyến nghị các quốc gia cần tăng cường hệ thống cảnh báo sớm để bảo vệ các nhóm dân cư dễ bị tổn thương. 
            Quan điểm ứng dụng công nghệ AI trong dự báo nhiệt độ cũng đồng nhất với các khuyến nghị từ những tổ chức quốc tế.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Tuy nhiên, các hệ thống dự báo nhiệt độ hiện nay tại Việt Nam vẫn còn nhiều hạn chế, chủ yếu dựa trên các phương pháp dự đoán kiểu truyền thống như hồi quy tuyến tính, ARIMA hoặc mô hình vật lý số (Numerical Weather Prediction - NWP), vốn đòi hỏi điều kiện của dữ liệu đầu vào rất cao. 
            Trong bối cảnh đó, trí tuệ nhân tạo (AI), đặc biệt là các mô hình học sâu hiện đại như Temporal Fusion Transformer (TFT), N-BEATS,... đang mở ra một hướng tiếp cận mới trong lĩnh vực dự báo thời tiết và nhiệt độ. 
            Trong nghiên cứu của Guoce Feng, và cộng sự, (2022) <b style="color:blue">[6]</b> , mô hình TFT cải tiến đã thể hiện khả năng vượt trội trong việc xử lý các chuỗi thời gian phức tạp nhờ khả năng xử lý lượng dữ liệu lớn và phi tuyến tính. 
            Cụ thể, khi áp dụng trong bài toán dự báo nhiệt độ không khí cung cấp trong toa tàu cao tốc, mô hình TFT cải tiến đã đạt kết quả cao về độ chính xác MAPE giảm 21,7% so với mô hình ban đầu và vượt trội so với các phương pháp truyền thống.
        </div>
        ''')
def two():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Với vai trò là nhóm sinh viên chuyên ngành Hệ thống thông tin, nhóm nhận thấy độ chính xác của các mô hình truyền thống hiện đang được sử dụng nhiều ở Việt Nam để dự báo nhiệt độ không còn đáng tin cậy trong bối cảnh Trái Đất đang ngày càng nóng lên do hiệu ứng nhà kín và biến đổi khí hậu ngày càng phức tạp. 
            Vì vậy để đáp ứng nhu cầu dự đoán nhiệt độ cực đại ngày càng tăng với độ chính xác cao, cần học hỏi và tiếp cận các mô hình học sâu mới hơn, hiện đại hơn, chính xác hơn. 
            Bên cạnh đó, qua quá trình nghiên cứu tài liệu, nhóm nhận thấy mặc dù đã có nhiều nghiên cứu quốc tế về ứng dụng mô hình học sâu trong dự báo thời tiết, nhưng vẫn chưa có nghiên cứu toàn diện nào đánh giá cụ thể hiệu quả của các mô hình học sâu trong những khu vực khác nhau với điều kiện khí hậu đặc thù của Việt Nam – một vấn đề cần được quan tâm và đầu tư nghiên cứu nhiều hơn.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {one() if name == "one" else ""}
            {two() if name == "two" else ""}
        </div>
        ''')