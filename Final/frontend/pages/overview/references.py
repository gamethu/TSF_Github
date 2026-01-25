import streamlit as st

def display():
    return (
        '''
        <div style="text-align: justify;">
            <b style="color:blue">[1]</b> &nbsp; A. M. Vicedo-Cabrera et al., "The burden of heat-related mortality attributable to recent human-induced climate change," Nature climate change, vol. 11, no. 6, pp. 492–500, 2021.<br>
            <b style="color:blue">[2]</b> &nbsp; W. B. Group, "Vietnam Country Climate and Development Report," ed: World Bank, 2022.<br>
            <b style="color:blue">[3]</b> &nbsp; B. Shamshad, M. Z. Khan, and Z. Omar, "Modeling and forecasting weather parameters using ANN-MLP, ARIMA and ETS model: a case study for Lahore, Pakistan," Journal of Applied Statistics, vol. 5, no. 388, p. 388, 2019.<br>
            <b style="color:blue">[4]</b> &nbsp; Phê duyệt Chiến lược quốc gia về biến đổi khí hậu giai đoạn đến năm 2050, Quyết định số 896/QĐ-TTg, 2022.<br>
            <b style="color:blue">[5]</b> &nbsp; S. Legg, "IPCC, 2021: Climate change 2021-the physical science basis," Interaction, vol. 49, no. 4, pp. 44–45, 2021.<br>
            <b style="color:blue">[6]</b> &nbsp; G. Feng, L. Zhang, F. Ai, Y. Zhang, and Y. Hou, "An improved temporal fusion transformers model for predicting supply air temperature in high-speed railway carriages," Entropy, vol. 24, no. 8, p. 1111, 2022.<br>
            <b style="color:blue">[7]</b> &nbsp; N. Chien et al., "ASSESSMENT OF CHANGE ON THE DAILY MAXIMUM HEAT INDEX FOR THAI BINH CITY (VIETNAM)," Applied Ecology & Environmental Research, vol. 22, no. 2, 2024.<br>
            <b style="color:blue">[8]</b> &nbsp; D. HUNG, C. PHUONG, N. THANG, N. CHIEN, B. DUNG, and V. HANG, "ASSESSING DAILY MAXIMUM HEAT INDEX IN THE CONTEXT OF CLIMATE CHANGE FOR HANOI (VIETNAM)," Applied Ecology & Environmental Research, vol. 22, no. 5, 2024.<br>
            <b style="color:blue">[9]</b> &nbsp; T. L. Hoang et al., "Assessing heat index changes in the context of climate change: a case study of Hanoi (Vietnam)," Frontiers in Earth Science, vol. 10, p. 897601, 2022.<br>
            <b style="color:blue">[10]</b> &nbsp; N. X. Trinh, T. Q. Trinh, T. P. Phan, T. N. Thanh, and B. N. Thanh, "Water temperature prediction models in northern coastal area, Vietnam," Asian Review of Environmental and Earth Sciences, vol. 6, no. 1, pp. 1–8, 2019.<br>
            <b style="color:blue">[11]</b> &nbsp; G. Y. Yun et al., "Predicting the magnitude and the characteristics of the urban heat island in coastal cities in the proximity of desert landforms. The case of Sydney," Science of The Total Environment, vol. 709, p. 136068, 2020.<br>
            <b style="color:blue">[12]</b> &nbsp; H. Jdi and N. Falih, "Leveraging transformer models for enhanced temperature forecasting: a comparative analysis in the Beni Mellal region," Indonesian Journal of Electrical Engineering and Computer Science, vol. 36, pp. 1694–1700, 2024.
        </div>
        ''')
def content(name):
    return (
       f'''
        <div style="text-align: justify">
            {display() if name == "all" else ""}
        </div>
        ''')