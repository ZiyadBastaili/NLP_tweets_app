import streamlit as st
import base64
from do_data.data_loader import DataLoader

class Application():
    def __init__(self):
        st.set_page_config(page_title='Tweet Visualization and Sentiment Analysis in Python', layout= "wide")

        st.markdown(""" <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style> """, unsafe_allow_html=True)

        padding = 3
        st.markdown(f""" <style>
                    .reportview-container .main .block-container{{
                        padding-top: {padding}rem;
                        padding-right: {padding}rem;
                        padding-left: {padding}rem;
                        padding-bottom: {padding}rem;
                    }} </style> """, unsafe_allow_html=True)

    def run_app(self):
        self.frame()

    def frame(self):
        #self.add_image("data/images/box_analytics.jpg")
        self.title()
        self.body()
        self.footer()

    def title(self):
        st.image("data/images/background.png", use_column_width=True)

    def body(self):
        #st.markdown("<h3 style='text-align: left; color: gry;font-family:courier;'> upload a csv </h3>", unsafe_allow_html=True)
        st.markdown("<h1> <br><br>  </h1>", unsafe_allow_html=True)
        DataLoader().read_data()

    def footer(self):
        st.markdown('<i style="font-size:11px">alpha version 0.1</i>', unsafe_allow_html=True)


    def add_image(self, url_image):
        LOGO_IMAGE = url_image

        st.markdown(
            """
            <style>
            .container {
                display: flex;
            }
            .logo-img {
                float:right;
            }
            img {
                display: block;
                width: 60%;
                margin-left: auto;
                margin-right: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.sidebar.markdown(
            f"""
            <div class="container">
                <div style="text-align: center;">
                <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
                </div>
            </div>
            <br><br>
            """,
            unsafe_allow_html=True
        )
