import streamlit as st

def display():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong bối cảnh biến đổi khí hậu toàn cầu đang diễn ra ngày càng phức tạp, hiện tượng nắng nóng cực đoan và nhiệt độ trung bình hàng ngày có xu hướng tăng cao, đặc biệt tại các khu vực ven biển và đô thị. 
            Thực tế này đã và đang gây ra những ảnh hưởng nghiêm trọng đến đời sống dân sinh, kinh tế - xã hội, cũng như đặt ra nhiều thách thức cho công tác phòng chống thiên tai và bảo vệ sức khỏe cộng đồng. 
            Chính vì vậy, việc dự báo nhiệt độ và cảnh báo sớm các đợt nắng nóng trở thành một trong những nội dung trọng tâm được các nhà khoa học, cơ quan chuyên môn và tổ chức nghiên cứu trong và ngoài nước đặc biệt quan tâm.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <b><i>* Tình hình nghiên cứu trong nước:</i></b>
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong những năm gần đây, đã có nhiều đề tài, dự án nghiên cứu được triển khai nhằm dự báo nhiệt độ cực đại, phân tích các yếu tố tác động đến hiện tượng nắng nóng, cũng như ứng dụng các mô hình học máy và học sâu trong lĩnh vực khí tượng thủy văn. 
            Các công trình này bước đầu đem lại những kết quả tích cực, song vẫn còn tồn tại một số hạn chế nhất định về phạm vi nghiên cứu, khả năng ứng dụng thực tiễn và độ chính xác trong điều kiện dữ liệu khí tượng của Việt Nam.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <i>Bài viết đăng trên các tạp chí khoa học, kỷ yếu hội thảo khoa học</i>
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong nghiên cứu NQ Chien, và cộng sự, (2024) <b style="color:blue">[7]</b>, các tác giả đã sử dụng mô hình hồi quy tuyến tính đa biến (Multivariable Linear Regression - MLR) để xây dựng phương trình nội suy giá trị độ ẩm tương đối tối thiểu hàng ngày (〖RH〗_min) dựa trên các yếu tố khí tượng gồm nhiệt độ tối đa, nhiệt độ tối thiểu và nhiệt độ trung bình hàng ngày tại Thái Bình, Việt Nam trong giai đoạn 1991-2021. 
            Kết quả nghiên cứu cho thấy rằng mô hình hồi quy hoạt động có hệ số xác định (R^2) đạt 0.6. 
            Sau đó, chỉ số nhiệt tối đa hàng ngày (〖HI〗_max) được tính toán từ giá trị T_maxvà 〖RH〗_min theo phương pháp của National Weather Service. 
            Mô hình cho kết quả hiệu suất rất cao với NSE đạt 0.95 trong giai đoạn hiệu chỉnh và 0.98 trong giai đoạn kiểm định. 
            Khi dự báo cho giai đoạn 2024-2054 theo hai kịch bản RCP 4.5 và RCP 8.5, chỉ số 〖HI〗_maxtrung bình năm có xu hướng tăng với hệ số xác định đạt lần lượt là 0.89 và 0.93, cho thấy độ tin cậy cao của xu thế tăng nhiệt trong tương lai.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Trong một số các nghiên cứu khác, các tác giả DN HUNG, và cộng sự, (2024) <b style="color:blue">[8]</b>, Thuy LT Hoang, và cộng sự, (2022) <b style="color:blue">[9]</b> cũng đã sử dụng mô hình hồi quy tuyến tính đa biến để xây dựng phương trình nội suy giá trị độ ẩm tương đối tối thiểu hàng ngày (RHmin) dựa trên các yếu tố khí tượng gồm nhiệt độ tối đa, nhiệt độ trung bình và lượng mưa hàng ngày tại thủ đô Hà Nội, Việt Nam trong giai đoạn 1991–2021. Sau đó, chỉ số nhiệt tối đa hàng ngày (〖HI〗_max) được tính toán từ giá trị T_maxvà 〖RH〗_min.
            Kết quả nghiên cứu cho thấy mô hình có hiệu suất khá tốt với hệ số xác định đạt 0.56. Giá trị NSE khi so sánh 〖HI〗_max tính toán với 〖HI〗_max thực đo trong giai đoạn hiệu chỉnh và kiểm định đều trên 0.94. 
            Đặc biệt, xu hướng tăng của HImax trung bình năm theo kịch bản biến đổi khí hậu được đánh giá có độ tin cậy rất cao với hệ số xác định đạt 0.89 đối với RCP 4.5 và 0.86 đối với RCP 8.5 trong giai đoạn 2021–2050, cho thấy rõ xu thế nóng lên và gia tăng nguy cơ nắng nóng tại khu vực này.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Bên cạnh đó, trong những năm gần đây, mạng nơ-ron nhân tạo (Artificial Neural Networks - ANN) đã và đang được ứng dụng khá rộng rãi trong nhiều lĩnh vực liên quan đến ước lượng và dự báo, đặc biệt đối với các mô hình phi tuyến phức tạp mà phương trình toán học truyền thống khó mô tả chính xác. 
            Nghiên cứu của Nguyen Xuan Trinh, và cộng sự, (2019) <b style="color:blue">[10]</b> đã tiến hành so sánh hiệu quả giữa các mô hình hồi quy tuyến tính (Linear Regression - LR), hồi quy phi tuyến (Non-linear Regression - NLR), hồi quy ngẫu nhiên (Stochastic Regression - SR) và mạng nơ-ron nhân tạo (ANN) trong dự báo nhiệt độ nước tối đa hàng ngày tại trạm Bãi Cháy, Quảng Ninh, Việt Nam dựa trên các biến khí tượng như nhiệt độ không khí tối đa, trung bình và các giá trị trễ theo thời gian trong giai đoạn 2008–2017. 
            Kết quả cho thấy, mô hình hồi quy tuyến tính mặc dù dễ xây dựng và giải thích nhưng có độ chính xác thấp với R = 0,89; RMSE = 1,68 và E = 0,89, chưa đạt ngưỡng tốt. 
            Mô hình hồi quy phi tuyến với hàm logistic cải thiện nhẹ (R = 0,9; RMSE = 1,6; E = 0,9) nhưng vẫn còn hạn chế. 
            Trong khi đó, mô hình hồi quy ngẫu nhiên kết hợp độ trễ thời gian 1 và 2 ngày đã nâng cao đáng kể hiệu quả dự báo với (R = 0,94; RMSE = 1,4; E = 0,92). 
            Đáng chú ý nhất, ANN với cấu trúc MLP4 sử dụng các biến T_max, T_mean, T_mean (t-1)và T_mean (t-2) cho kết quả tốt nhất với (R = 0,988; RMSE = 1,24; E = 0,94), khẳng định khả năng xử lý các quan hệ phi tuyến và các đặc điểm động học nhiệt của nước tốt hơn hẳn so với các phương pháp hồi quy truyền thống. 
            Kết quả nghiên cứu cho thấy ANN có thể loại bỏ yêu cầu giả định phân phối chuẩn và tuyến tính của các biến, phù hợp với các bài toán dự báo khí tượng thủy văn phức tạp và có tính phi tuyến cao, đặc biệt trong bối cảnh biến đổi khí hậu ngày càng khó lường hiện nay.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Tuy nhiên, có thể nhận thấy rằng phần lớn các nghiên cứu hiện nay vẫn chủ yếu ứng dụng các mô hình thống kê truyền thống hoặc các phương pháp nội suy bằng phương trình toán học dựa trên các yếu tố khí tượng cơ bản như nhiệt độ, độ ẩm và lượng mưa. 
            Các mô hình này mặc dù dễ xây dựng, dễ hiệu chỉnh và có thể giải thích trực quan, song chưa khai thác hiệu quả mối quan hệ phức tạp và phi tuyến giữa các đặc trưng đầu vào. 
            Đặc biệt, các yếu tố tự nhiên có ảnh hưởng đáng kể đến nhiệt độ và chỉ số nhiệt như đặc điểm địa hình (kinh độ, vĩ độ, độ cao), hướng gió, độ che phủ mây, cũng như điều kiện khu vực (gần biển, đô thị hóa) lại chưa được tích hợp đầy đủ vào các mô hình dự báo hiện có. 
            Bên cạnh đó, phạm vi nghiên cứu trong các công trình khảo sát còn tương đối hẹp, chủ yếu giới hạn ở cấp tỉnh hoặc thành phố đơn lẻ như Thái Bình, Hà Nội hay Quảng Ninh. Việc mở rộng các nghiên cứu quy mô liên vùng, đặc biệt tại các đô thị ven biển và các khu vực nhạy cảm với nắng nóng và biến đổi khí hậu trên phạm vi quốc gia hoặc khu vực Đông Nam Á vẫn còn rất hạn chế. 
            Điều này dẫn đến tính ứng dụng thực tiễn và khả năng tổng quát hóa của kết quả nghiên cứu chưa thực sự cao. 
            Ngoài ra, dù một số nghiên cứu gần đây đã bước đầu tiếp cận các mô hình học sâu như mạng nơ-ron nhân tạo (ANN), song số lượng còn ít, chưa đa dạng về kiến trúc mô hình, phương pháp hiệu chỉnh tham số, cũng như đánh giá thực nghiệm về hiệu quả so sánh với các thuật toán học máy khác (Random Forest, XgBoost, LSTM…).
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <b><i>* Tình hình nghiên cứu nước ngoài:</i></b>
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Tại các quốc gia phát triển và một số nước khu vực châu Á, việc dự báo nhiệt độ cực đại và cảnh báo nắng nóng bằng các mô hình chuỗi thời gian hiện đại đã được chú trọng từ sớm. 
            Đặc biệt, các mô hình học sâu như mô hình bộ nhớ dài ngắn hạn (Long Short-Term Memory - LSTM), Gated Recurrent Unit (GRU) và Temporal Fusion Transformer (TFT) đang được áp dụng rộng rãi cho dự báo nhiệt độ, lượng mưa và các chỉ số thời tiết cực đoan.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Nghiên cứu của Geun Young Yun, và cộng sự, (2020) <b style="color:blue">[11]</b> tại thành phố Sydney, Úc là một trong những công trình tiên phong ứng dụng mô hình Long Short-Term Memory (LSTM) để dự báo cường độ đảo nhiệt đô thị (Urban Heat Island Intensity - UHII) với bộ dữ liệu nhiệt độ không khí hàng giờ thu thập liên tục trong 18 năm (1999–2017). 
            Kết quả cho thấy LSTM có khả năng dự báo chính xác UHII ngay cả trong điều kiện nhiệt độ biến động mạnh do chịu tác động đồng thời của hai hệ thống khí lưu đối lập là gió biển và gió sa mạc — yếu tố mà các mô hình thống kê truyền thống thường gặp khó khăn khi xử lý.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Nghiên cứu của Hamza Jdi and Noubluedine Falih, (2024) <b style="color:blue">[12]</b> tại Morocco đã ứng dụng mô hình Temporal Fusion Transformer (TFT) để dự báo nhiệt độ trung bình ngày, với chuỗi dữ liệu kéo dài 38 năm (1984 - 2022). 
            Nghiên cứu tiến hành so sánh hiệu quả giữa TFT với các mô hình truyền thống và học sâu phổ biến gồm ARIMA, Simple RNN, GRU và LSTM. 
            Kết quả cho thấy TFT có độ chính xác vượt trội, với giá trị Mean Absolute Error (MAE) chỉ 1.5143 và hệ số xác định đạt 0.9359, cao hơn rõ rệt so với các mô hình còn lại. 
            TFT cũng chứng minh khả năng xử lý tốt các mối quan hệ dài hạn và các đặc điểm bất thường trong chuỗi thời gian - điểm hạn chế của các mô hình RNN truyền thống và ARIMA.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            Nhìn chung, các nghiên cứu trong nước chủ yếu sử dụng các mô hình hồi quy thống kê truyền thống với dữ liệu đầu vào hạn chế, cho kết quả ở mức khá nhưng còn yếu trong xử lý các quan hệ phi tuyến và biến động bất thường. 
            Các nghiên cứu quốc tế như của Geun Young Yun, và cộng sự, (2020) <b style="color:blue">[11]</b>, Hamza Jdi and Noubluedine Falih, (2024) <b style="color:blue">[12]</b> ứng dụng các mô hình học sâu tiên tiến hơn như LSTM và TFT, đạt độ chính xác vượt trội với hệ số xác định trên 0.93 và khả năng dự báo tốt các biến động dài hạn. 
            Tuy nhiên, điểm hạn chế chung của cả hai nhóm là chưa tích hợp đầy đủ các yếu tố địa hình, độ che phủ mây, tốc độ gió,… và phạm vi nghiên cứu còn hẹp, chưa mở rộng sang các khu vực ven biển - nơi chịu ảnh hưởng nặng nề của nắng nóng và biến đổi khí hậu. Điều này mở ra cơ hội cho việc phát triển các mô hình học sâu tích hợp yếu tố môi trường tại Việt Nam trong thời gian tới.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {display() if name == "all" else ""}
        </div>
        ''')