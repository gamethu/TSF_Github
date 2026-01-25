import streamlit as st

def one():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            - Tìm hiểu và xây dựng mô hình, thuật toán phù hợp để dự đoán chính xác nhiệt độ tối đa hàng ngày ở một số khu vực tỉnh thành Việt Nam. 
        </div>
        ''')
def two():
    return (
        '''
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            - Hệ thống hoá cơ sở lý luận và kinh nghiệm thực tế trong việc xây dựng mô hình chuỗi thời gian dự đoán nhiệt độ cực đại.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            - Đánh giá độ chính xác của mô hình chuỗi thời gian dự đoán nhiệt độ cực đại, chỉ ra những điểm đã làm tốt, chưa làm tốt và nguyên nhân.
        </div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            - Đưa ra hệ thống các giải pháp toàn diện và khả thi dựa trên dự đoán của mô hình chuỗi thời gian dự đoán nhiệt độ cực đại và tình trạng chung của khu vực được khảo sát nhằm đưa ra cảnh báo về các đợt nắng nóng sắp diễn ra.
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