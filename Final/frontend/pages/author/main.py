import streamlit as st
import pandas as pd


def display():
    data = {
    "Họ và tên": [
        "Nguyễn Tấn Đại",
        "Nguyễn Tấn Đại",
        "Nguyễn Tấn Đại",
    ],
    "SĐT": [
        "0908693078",
        "0908693078",
        "0908693078",
    ],
    "Email cá nhân": [
        "nguyentandai.nckh@gmail.com",
        "nguyentandai.nckh@gmail.com",
        "nguyentandai.nckh@gmail.com",
    ],
    "MSSV": [
        "3122411035",
        "3122411035",
        "3122411035",
    ],
    "Chuyên ngành": [
        "Công nghệ thông tin",
        "Công nghệ thông tin",
        "Công nghệ thông tin",
    ],
    "Khoa": [
        "Công nghệ thông tin",
        "Công nghệ thông tin",
        "Công nghệ thông tin",
    ],
    "Trường": [
        "Đại học Sài Gòn",
        "Đại học Sài Gòn",
        "Đại học Sài Gòn",
    ],
    "Năm học": [4, 4, 4],
    "Email sinh viên": [
        "3122411035@sv.sgu.edu.vn",
        "3122411035@sv.sgu.edu.vn",
        "3122411035@sv.sgu.edu.vn",
    ],}
    
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Họ và tên": st.column_config.TextColumn(width="medium"),
            "Email cá nhân": st.column_config.TextColumn(width="large"),
            "Email sinh viên": st.column_config.TextColumn(width="large"),
            "Trường": st.column_config.TextColumn(width="medium"),
            "Năm học": st.column_config.NumberColumn(width="small"),
        }
    )
display()