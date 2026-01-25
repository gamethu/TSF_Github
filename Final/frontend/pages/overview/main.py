import streamlit as st
from pages.overview import (one,
                            two,
                            three,
                            four,
                            six,
                            seven,
                            eight,
                            references)
def side_bar():
    with st.sidebar:
        st.markdown(
        """
        <h4 style="text-align:center">MỤC LỤC</h4>

        <ul style="list-style-type: none">
            <li>
                <a href="#1">1. Lý do chọn đề tài</a>
                <ul style="list-style-type: none">
                    <li><a href="#11">1.1. Bối cảnh thực tiễn</a></li>
                    <li><a href="#12">1.2. Bối cảnh lý thuyết</a></li>
                </ul>
            </li>
            <li>
                <a href="#2">2. Tổng quan nghiên cứu</a>
            </li>
            <li>
                <a href="#3">3. Mục tiêu nghiên cứu</a>
                <ul style="list-style-type: none">
                    <li><a href="#31">3.1. Mục tiêu chung</a></li>
                    <li><a href="#32">3.2. Mục tiêu riêng</a></li>
                </ul>
            </li>
            <li>
                <a href="#4">4. Nhiệm vụ nghiên cứu</a>
            </li>
            <li>
                <a href="#6">6. Phạm vi nghiên cứu</a>
                <ul style="list-style-type: none">
                    <li><a href="#61">6.1. Phạm vi không gian</a></li>
                    <li><a href="#62">6.2. Phạm vi thời gian</a></li>
                    <li><a href="#63">6.3. Phạm vi nội dung</a></li>
                </ul>
            </li>
            <li>
                <a href="#7">7. Phương pháp nghiên cứu</a>
            </li>
            <li>
                <a href="#8">8. Ý nghĩa thực tiễn của đề tài</a>
            </li>
        </ul>
        
        <h4 style="text-align:center">DANH MỤC TÀI LIỆU THAM KHẢO</h4>
        """,
        unsafe_allow_html=True
    )
def content():
    st.markdown(
       f'''
        <div id='1'>
            <h4>1. Lý do chọn đề tài</h4>
        </div>
        <div id='11'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>1.1 Bối cảnh thực tiễn</i></h5></div> {one.content("one")}
        <div id='12'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>1.2 Bối cảnh lý thuyết</i></h5></div> {one.content("two")}
        <div id='2'>
            <h4>2. Tổng quan nghiên cứu</h4>
        </div>
        {two.content("all")}
        <div id='3'>
            <h4>3. Mục tiêu nghiên cứu</h4>
        </div>
        <div id='31'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>3.1. Mục tiêu chung</i></h5></div> {three.content("one")}
        <div id='32'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>3.2. Mục tiêu riêng</i></h5></div> {three.content("two")}
        <div id='4'>
            <h4>4. Nhiệm vụ nghiên cứu</h4>
        </div>
        {four.content("all")}
        <div id='6'>
            <h4>6. Phạm vi nghiên cứu</h4>
        </div>
        <div id='61'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>6.1. Phạm vi không gian</i></h5></div> {six.content("one")}
        <div id='62'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>6.2. Phạm vi thời gian</i></h5></div>  {six.content("two")}
        <div id='62'><h5>&nbsp;&nbsp;&nbsp;&nbsp;<i>6.3. Phạm vi nội dung</i></h5></div>   {six.content("three")}
        <div id='7'>
            <h4>7. Phương pháp nghiên cứu</h4>
        </div>
        {seven.content("all")}
        <div id='8'>
            <h4>8. Ý nghĩa thực tiễn của đề tài</h4>
        </div>
        {eight.content("all")}
        <div id='ref'>
            <h4 style="text-align:center">DANH MỤC TÀI LIỆU THAM KHẢO</h4>
        </div>
        {references.content("all")}
        ''', 
        unsafe_allow_html=True)
    
def display():
    side_bar()
    content()
    
display()