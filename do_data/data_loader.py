import streamlit as st
from st_annotated_text import annotated_text
import pandas as pd
import re
import requests
import numpy as np
import datetime
import twint
import asyncio
import nest_asyncio  #ggg
from googletrans import Translator
import texthero as hero
from texthero import preprocessing
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import ast
from itertools import product
import itertools
import seaborn as sns
import plotly.graph_objects as go




# import docx2txt
# from PyPDF2 import PdfFileReader
# import pdfplumber

import time
import gc
import base64
import io
import pickle

from setuptools.command.setopt import option_base
from textblob import TextBlob

from collections import defaultdict

from streamlit.report_thread import get_report_ctx
from streamlit import caching

class DataLoader():
    def __init__(self):
        self.test = None
        self.upload_file = None
        self.df = None
        self.tweet_df = None
        self.df_edit = None
        self.index_col = None
        self.select_col = None
        self.select_dataType = None
        self.usecols = None
        self.set_dtypes = None
        self.refresh = None
        self.usedetails = list()
        self.More_details = None
        self.followers = None

    def Sentiment_Analysis(self, mode):
        st.session_state.test = mode

    def read_data(self):

        # open and read the file after the appending:
        # f = open("data/test.txt", "r")
        # self.test = str(f.read())
        # print("===============>"+ self.test)
        if 'df' not in st.session_state:
            st.session_state.df = self.tweet_df

        if 'test' not in st.session_state:
            st.session_state.test = 'Data Processing web page'

        if st.session_state.test == 'Data Processing web page':
            st.title('Tweets Scraping and Processing')

            self.choose_dataType()
            if self.select_dataType == 'twitter':
                self.upload_file = None

                self.More_details = st.checkbox('More details on Tweets', value=False, key='5')
                if self.More_details:
                    self.select_details()
                self.more_details()


            else:
                label='upload a csv'
                type=['csv','xlsx']
                accept_multiple_files=False
                key='dataloader1'

                self.upload_file = st.file_uploader(label=label
                                               , type=type
                                               , accept_multiple_files=accept_multiple_files
                                               , key=key)

            if self.upload_file is not None:
                self.edit_data()

                with st.spinner('Reading and Parsing DataFrame Columns'):
                    try:
                        self.tweet_df = pd.read_csv(self.upload_file)
                    except:
                        self.tweet_df = pd.read_excel(self.upload_file)
                    self.tweet_df.columns = self.parse_cols(self.tweet_df)
                    # if data come from excel file, the hashtag field is a string not a list so we have to convert it
                    # Converting string to list ---------------------------------------
                    self.tweet_df['hashtags'] = self.tweet_df['hashtags'].apply(lambda str_list: ast.literal_eval(str_list))

                st.success('Ready! Parsed DataFrame Columns to lower case with underscores _.')

                if self.select_col:
                    self.select_cols()

                st.markdown("<h3 style='text-align: center; color: gry;'> File Preview </h3>", unsafe_allow_html=True)
                if self.df_edit is not None:
                    st.write(self.df_edit)
                else:
                    st.write(self.tweet_df)

            if self.tweet_df is not None:

                # Standardize the language
                st.markdown("<h2 style='text-align: left; color: gry;'> Standardize the language </h2>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: left; color: gry;'> Language of Tweets : &emsp; &emsp;<span style='color: yellow;'>" + str(self.tweet_df['language'].unique()) +"<span/></h4><br>",
                            unsafe_allow_html=True)
                test = 0
                L = len(self.tweet_df)
                if self.select_dataType == 'csv':
                    actions = ['Select Action: ','Delete all tweets that are not in English', 'Translate all tweets that are not in English']
                    my_expander = st.expander("Select Action", expanded=True)
                    with my_expander:
                        #st.markdown("<h4 style='text-align: left; color: gry;'> <br>Select Action :</h4>",unsafe_allow_html=True)
                        make_choice = st.selectbox('', actions, index=actions.index('Select Action: '))

                    if make_choice == 'Delete all tweets that are not in English':
                        self.tweet_df = self.tweet_df[self.tweet_df.language == 'en']
                        st.success('Delete all tweets that are not in English... Done')
                        test=1

                    elif make_choice == 'Translate all tweets that are not in English':

                        translator = Translator()
                        pr = st.text('Translation... ' + str(self.porcentage(0, L)))
                        for i in range(0, L):
                            pr.text('Translation... ' + str(self.porcentage(i,L)))
                            if self.tweet_df.iloc[i]['language'] != 'en':
                                try:
                                    self.tweet_df.loc[i, 'tweet'] = translator.translate(self.tweet_df.iloc[i]['tweet'], src=self.tweet_df.iloc[i]['language'], dest='en').text
                                except:
                                    try:
                                        self.tweet_df.loc[i, 'tweet'] = translator.translate(self.tweet_df.iloc[i]['tweet'], dest='en').text
                                    except:
                                        self.tweet_df.loc[i, 'tweet'] = self.tweet_df.iloc[i]['tweet']
                            else:
                                self.tweet_df.loc[i, 'tweet'] = self.tweet_df.iloc[i]['tweet']
                        pr.text('Translation... ' + str(self.porcentage(L, L)))
                        st.success('Translate all tweets that are not in English... Done')
                        test = 1
                else:
                    self.tweet_df = self.tweet_df[self.tweet_df.language == 'en']
                    st.success('Delete all tweets that are not in English... Done')
                    test = 1

                if test==1:
                    # dropping ALL duplicate values
                    st.markdown("<h2 style='text-align: left; color: gry;'> Dropping ALL duplicate values </h2>", unsafe_allow_html=True)
                    self.tweet_df.drop_duplicates(subset="tweet", keep=False, inplace=True)
                    st.success('Initial tweets count: `%s`\n\nFinal tweets count:`%s`' % (L, len(self.tweet_df)))

                    # Data preprocessing
                    st.markdown("<h2 style='text-align: left; color: gry;'> Data preprocessing </h2>", unsafe_allow_html=True)

                    with st.spinner('Tweets are being cleaned...'):
                        # just in case texthero cant remove URLs
                        st.subheader('Remove urls')
                        self.tweet_df['clean_tweet'] = self.tweet_df['tweet'].str.replace('http\S+|www.\S+', '', case=False)
                        st.success('Remove urls... Done')
                        st.table(self.tweet_df[['tweet','clean_tweet']].head(5))

                        # remove mentions
                                #self.tweet_df['tweet'] = self.tweet_df['tweet'].apply(lambda x: re.sub(r'\@\w+','',x))


                        # remove punctuation
                        st.subheader('Remove punctuation')
                                # creating a custom pipeline to preprocess the raw text we have
                                # custom_punctuation_pipeline = [preprocessing.fillna
                                #     #, preprocessing.lowercase
                                #     #, preprocessing.remove_digits
                                #     # , preprocessing.remove_whitespace
                                #     , preprocessing.remove_punctuation]
                                # # simply call clean() method to clean the raw text in 'tweet' col and pass the custom_pipeline to pipeline argument
                                # self.tweet_df['clean_tweet'] = hero.clean(self.tweet_df['clean_tweet'], pipeline=custom_punctuation_pipeline)
                        self.tweet_df['clean_tweet'] = self.tweet_df['clean_tweet'].apply(self.remove_punctuation, args=(''''!"&\'()*+,-./:;<=>?-[\\]^_{|}~`''',))
                        st.success('Remove punctuation... Done')
                        st.table(self.tweet_df[['tweet', 'clean_tweet']].head(5))

                        # Remove diacritics
                        st.subheader('Remove diacritics')
                        custom_diacritics_pipeline = [preprocessing.remove_diacritics]
                        self.tweet_df['clean_tweet'] = hero.clean(self.tweet_df['clean_tweet'], pipeline=custom_diacritics_pipeline)
                        st.success('Remove diacritics... Done')
                        st.table(self.tweet_df[['tweet', 'clean_tweet']].head(5))

                        # remove stopwords
                        st.subheader('Remove stopwords')
                        custom_stopwords_pipeline = [preprocessing.remove_stopwords]
                        self.tweet_df['clean_tweet'] = hero.clean(self.tweet_df['clean_tweet'], pipeline=custom_stopwords_pipeline)
                        st.success('Remove stopwords... Done')
                        st.table(self.tweet_df[['tweet', 'clean_tweet']].head(5))

                        # remove digits
                        st.subheader('Remove digits')
                        self.tweet_df['clean_tweet'] = self.tweet_df['clean_tweet'].apply(
                            lambda x: re.sub(r'\s\d+\s', ' ', x))
                        st.success('Remove digits... Done')
                        st.table(self.tweet_df[['tweet', 'clean_tweet']].head(5))


                        # stemming
                        st.subheader('stemming')
                        custom_stem_pipeline = [preprocessing.stem]
                        self.tweet_df['clean_tweet'] = hero.clean(self.tweet_df['clean_tweet'], pipeline=custom_stem_pipeline)
                        st.success('stemming... Done')
                        st.table(self.tweet_df[['tweet', 'clean_tweet']].head(5))


                    # Clean Data
                    st.markdown("<h2 style='text-align: left; color: gry;'> Clean Data </h2>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center; color: gry;'> Data Preview </h3>", unsafe_allow_html=True)
                    st.write(self.tweet_df[['username','tweet', 'clean_tweet','hashtags']])
                    st.session_state.df = self.tweet_df

                    # select top hashtags
                    st.markdown("<h2 style='text-align: left; color: gry;'> <br>Select Top Hashtags </h2>", unsafe_allow_html=True)

                    hashtags_list = self.hashtags_extract(self.tweet_df)
                    col1, col2 = st.columns(2)

                    with col1:
                        my_expander = st.expander("Hashtags List", expanded=False)
                        with my_expander:
                            st.subheader('Hashtags List')
                            freq = nltk.FreqDist(hashtags_list)
                            d = pd.DataFrame({'Hashtag': list(freq.keys()),'Count': list(freq.values())})

                            st.write(d.sort_values(by=['Count'], ascending=False).reset_index(drop=True))
                    with col2:
                        my_expander = st.expander("Top 10 Hashtags", expanded=False)
                        with my_expander:
                            st.subheader('Top 10 Hashtags')
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            dplot = d.nlargest(columns='Count', n=10)
                            plt.figure(figsize=(15, 10.7))
                            sns.barplot(data=dplot, x='Hashtag', y='Count')
                            st.pyplot()

                    # select Codes
                    st.markdown("<h2 style='text-align: left; color: gry;'> <br>Select Codes </h2>", unsafe_allow_html=True)

                    codes_list = self.codes_extract(self.tweet_df)
                    col1, col2 = st.columns((1,2))

                    with col1:
                        my_expander = st.expander("Codes List", expanded=False)
                        with my_expander:
                            st.subheader('Codes List')
                            freq = nltk.FreqDist(codes_list)
                            d = pd.DataFrame({'Codes': list(freq.keys()), 'Count': list(freq.values())})

                            st.write(d.sort_values(by=['Count'], ascending=False).reset_index(drop=True))
                    with col2:
                        my_expander = st.expander("Top 10 Codes", expanded=False)
                        with my_expander:
                            st.subheader('Top 10 Codes')
                            dplot = d.nlargest(columns='Count', n=10)
                            plt.figure(figsize=(15, 6.9))
                            sns.barplot(data=dplot, x='Codes', y='Count')
                            st.pyplot()




                    st.markdown('<br><br>', unsafe_allow_html=True)
                    col1, col2, col4, col5 = st.columns(4)
                    with col2:
                        st.button('Sentiment Analysis', on_click=self.Sentiment_Analysis, args=('Data Analysis web page',))
                    with col4:
                        st.button('User Profile Analysis', on_click=self.Sentiment_Analysis, args=('Users Analysis web page',))



        elif st.session_state.test == 'Users Analysis web page':
            self.local_css("style.css")

            self.tweet_df = st.session_state.df

            # select active comptes
            st.markdown("<h2 style='text-align: left; color: gry;'> Active comptes </h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                my_expander = st.expander("Average Likes Per Person", expanded=False)
                with my_expander:
                    st.subheader('Average Likes Per Person')
                    tweet_likes = self.tweet_df[['username', 'nlikes']]
                    st.write(tweet_likes.groupby(['username']).mean().sort_values(by=['nlikes'], ascending=False).reset_index())
            with col2:
                my_expander = st.expander("Number of Tweets Per Person", expanded=False)
                with my_expander:
                    st.subheader('Average Tweets Per Person')
                    tweet_count = self.tweet_df[['username']]
                    tweet_count = tweet_count.groupby('username').size().reset_index(name='Size')
                    st.write(tweet_count.sort_values(by=['Size'], ascending=False).reset_index(drop=True))

            border = "<div class='highlight red'></div>"
            st.markdown(border, unsafe_allow_html=True)

            # WordCloud
            tw = self.tweet_df[['username']]
            tw = tw.groupby('username').size().reset_index(name='Size')
            userList = tw['username']
            userList = userList.values.tolist() + [s.replace("@", "") for s in self.mentions_extract(self.tweet_df)]
            userSearch = list(dict.fromkeys(userList)) #remove duplicates users to know how many users exactly

            # mentions visualization
            st.markdown("<h2 style='text-align: left; color: gry;'> Word cloud for @USERS (Total : " + str(len(userSearch)) +" Users) </h2>",
                        unsafe_allow_html=True)
            words = ' '.join(userList)
            wordcloud = WordCloud(background_color='white', width=800, height=640).generate(words)
            wordcloud_fig = plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            st.pyplot()

            border = "<div class='highlight blue'></div>"
            st.markdown(border, unsafe_allow_html=True)

            st.markdown("<h2 style='text-align: left; color: gry;'> User Profile Analysis </h2>",
                        unsafe_allow_html=True)

            col1, sep, col2 = st.columns((1,0.2,1))
            with col1:
                st.markdown('<br>', unsafe_allow_html=True)
                self.USERS = st.selectbox(label="Select a username", options=userSearch, key='10')
                        #'thenizzar'
            with col2:
                with st.form(key='my_form'):
                    search_user = st.text_input("Searching for a User")
                    if st.form_submit_button(label='Submit') :
                        self.USERS = search_user



            st.markdown('<br><br>', unsafe_allow_html=True)

            if self.USERS != 'Select username:' and self.USERS is not None and self.USERS !='':
                try:
                    user_info = self.user_profile(self.USERS)
                    self.display_profile(user_info)

                    # transform the timestamp to datetime object
                    user_df = self.tweet_df[['username', 'tweet', 'nlikes', 'nreplies', 'nretweets', 'date']]
                    user_df['date'] = pd.to_datetime(user_df['date'], errors='coerce')
                    # extract dayofweek, month, week, year, ym
                    user_df['dayofweek'] = user_df['date'].dt.dayofweek
                    user_df['month'] = user_df['date'].dt.month
                    user_df['week'] = user_df['date'].dt.week
                    user_df['year'] = user_df['date'].dt.year
                    user_df['ym'] = user_df['year'].astype(str) + user_df['month'].astype(str)

                    engagement_rate = (((user_df['nlikes'].sum() + user_df['nretweets'].sum() + user_df['nreplies'].sum()) / len(user_df)) /
                                       self.followers) * 100
                    st.write(' ')
                    col1, col2 = st.columns(2)
                    col1.markdown(annotated_text(("Average Posts Per Month:", "", "#b3ffcc"), " ", str(round(user_df.groupby('ym').size().mean(), 2))), unsafe_allow_html=True)
                    col2.markdown(annotated_text(("Engagement Rate:", "", "#b3ffcc"), " ", str(round(engagement_rate, 2)) + '%'), unsafe_allow_html=True)

                    st.markdown('<br><br>', unsafe_allow_html=True)

                    # statistics
                    df_st = user_df.query('username=="'+self.USERS+'"')
                    x = df_st.groupby('dayofweek').size()
                    st.subheader('Number Of Posts Per Week-Day')
                    st.bar_chart(pd.DataFrame(x).rename(columns={0: 'Number Of Posts'}))

                    x = df_st.groupby('month').size()
                    st.subheader('Number Of Posts Per Month')
                    st.bar_chart(pd.DataFrame(x).rename(columns={0: 'Number Of Posts'}))

                except Exception as e:
                    col1, col2 = st.columns((1,2))
                    col1.image('data/images/ops_twitter/ops.png')
                    col2.write('')
                    col2.warning(e)

            # back
            st.sidebar.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.sidebar.columns((1,2,1))
            col2.button('Data Processing web page', on_click=self.Sentiment_Analysis, args=('Data Processing web page',))
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            col2.button('Sentiment Analysis', on_click=self.Sentiment_Analysis, args=('Data Analysis web page',))

        elif st.session_state.test == 'Data Analysis web page':
            # Data Analysis web page
            st.title('Sentiment Analysis of Tweets')
            st.markdown(" This application is a Streamlit dashboard to analyze the sentiment of Tweets...")

            # prepare dataframe
            # Create two new columns
            self.tweet_df = st.session_state.df
            self.tweet_df['Subjectivity'] = self.tweet_df['clean_tweet'].apply(self.getSubjectivity)
            self.tweet_df['Polarity'] = self.tweet_df['clean_tweet'].apply(self.getPolarity)
            self.tweet_df['Analysis'] = self.tweet_df['Polarity'].apply(self.getAnalysis)

            # d1=self.tweet_df[['tweet','clean_tweet','Subjectivity','Polarity','Analysis']].style.background_gradient(cmap='RdYlGn', low=0, high=0, axis=0, subset='Polarity')
            # st.subheader("1) Table of Tweets :")
            # st.write(d1)

            d1=self.tweet_df[['tweet','clean_tweet','Subjectivity','Polarity','Analysis']].style.apply(self.color, axis=None)
            st.write(d1)

            # data visualizations
            st.subheader("2) Visualizations :")
            st.sidebar.markdown("### Number of tweets by sentiment")
            # select box widget
            select = st.sidebar.selectbox("Visualization Type:", ['Histogram', 'Pie Chart', 'scatter'], key='2')
            sentiment_count = self.tweet_df['Analysis'].value_counts()
            # st.write(sentiment_count)
            sentiment_count_df = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

            # plots
            if select == "Histogram":
                fig_choice = px.histogram(self.tweet_df, x='Analysis', histfunc='count', color='Analysis', height=600, width=800, color_discrete_map={"Positive":"green", "Negative":"red", "Neutral":"#CCCC00"})
                st.plotly_chart(fig_choice)
            elif select == 'Pie Chart':
                fig = px.pie(sentiment_count_df, values='Tweets', names='Sentiment')
                st.plotly_chart(fig)
            else:
                g = sns.relplot(x="Polarity", y="Subjectivity", hue="Analysis", data=self.tweet_df)
                g.fig.subplots_adjust()
                st.pyplot()


            # creating word cloud visualizations
            st.sidebar.header("Word Cloud")
            word_sentiment = st.sidebar.radio('Display WordCloud for sentiment:', ('Positive', 'Neutral', 'Negative'), key='4')
            st.subheader('Word cloud for %s sentiment' % word_sentiment)
            df = self.tweet_df[self.tweet_df['Analysis'] == word_sentiment]
            words = ' '.join(df['clean_tweet'])
            wordcloud = WordCloud(background_color='white', width=800, height=640).generate(words)
            wordcloud_fig = plt.imshow(wordcloud, interpolation='bilinear')
            plt.xticks([])
            plt.yticks([])
            st.pyplot()



            # back
            st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.sidebar.columns(3)
            col2.button('Data Processing web page', on_click=self.Sentiment_Analysis, args=('Data Processing web page', ))


    def user_profile(self, username):
        c = twint.Config()
        c.Username = username
        c.Store_object = True
        c.Hide_output = True
        twint.run.Lookup(c)
        user_profile = twint.output.users_list[-1]
        return user_profile

    def display_profile(self, user_profile):
        # get the profile_pic_url
        prof_pic = user_profile.avatar.replace("normal", "400x400")

        # download the image in a folder called static I created
        response = requests.get(prof_pic)
        filename = "image.jpg"
        with open(filename, "wb") as f:
            f.write(response.content)

        # show the full name
        st.markdown(annotated_text(("Full Name:", "", "#fea")," ", user_profile.name), unsafe_allow_html=True)
        st.write(' ')
        # we can format the output into as many columns as we want using beta_columns
        col1, col2 = st.columns(2)
        col1.image(filename)
        col2.markdown(annotated_text(("Biography:", "", "#faa")," ", user_profile.bio), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Location:", "", "#faa")," ", user_profile.location), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Number Of Tweets:", "", "#faa")," ", str(user_profile.tweets)), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Number Of Following:", "", "#faa")," ", str(user_profile.following)), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Number Of Followers:", "", "#faa")," ", str(user_profile.followers)), unsafe_allow_html=True)
        self.followers = user_profile.followers
        col2.write(' ')
        col2.markdown(annotated_text(("Is Private Account:", "", "#faa")," ", str(user_profile.is_private)), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Is Verified Account:", "", "#faa")," ", str(user_profile.is_verified)), unsafe_allow_html=True)
        col2.write(' ')
        col2.markdown(annotated_text(("Number Of Liked tweets:", "", "#faa")," ", str(user_profile.likes)), unsafe_allow_html=True)
        col2.write(' ')
        date = str(user_profile.join_date) + str(' ') + str(user_profile.join_time)
        col2.markdown(annotated_text(("Date Of Join:", "", "#faa")," ", date), unsafe_allow_html=True)



    def color(self, df):
        c1 = 'background-color: green'
        c2 = 'background-color: red'
        c3 = 'background-color: #CCCC00'
        c = ''
        # compare columns
        mask1 = (df['Analysis'] == 'Positive')
        mask2 = (df['Analysis'] == 'Negative')
        mask3 = (df['Analysis'] == 'Neutral')
        # DataFrame with same index and columns names as original filled empty strings
        df1 = pd.DataFrame(c, index=df.index, columns=df.columns)
        # modify values of df1 column by boolean mask
        df1.loc[mask1, ['Polarity','Analysis']] = c1
        df1.loc[mask2, ['Polarity','Analysis']] = c2
        df1.loc[mask3, ['Polarity','Analysis']] = c3
        return df1


    def local_css(self, file_name):
        with open(file_name) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


    def read_test_from_file(self, test):
        f = open("data/test.txt", "w")
        f.write(str(test))
        f.close()

    def remove_punctuation(self, text, punctuations):
        """custom function to remove the punctuation"""
        return text.translate(str.maketrans('', '', punctuations))


    def porcentage(self, num, total):
        p = num / total
        percentage = "{:.2%}".format(p)
        return percentage

    # extract the Hashtags
    def hashtags_extract(self, tweets):
        hashtags = []
        # loop words in the tweet
        for i in tweets.index:
            hashtag = tweets['hashtags'][i]
            hashtags.append(hashtag)
        hashtag_list = list(itertools.chain.from_iterable(hashtags))
        hashtag_list = [elem.upper() for elem in hashtag_list]
        return hashtag_list

    # extract the codes
    def codes_extract(self, tweets):
        codes = []
        # loop words in the tweet
        for i in range(0, len(tweets)):
            code = re.findall(r'[$]\w+', tweets.iloc[i]['tweet'])
            codes.append(code)
        codes_list = list(itertools.chain.from_iterable(codes))
        codes_list = [elem.upper() for elem in codes_list]
        return codes_list

    # extract the mentions
    def mentions_extract(self, tweets):
        mentions = []
        # loop words in the tweet
        for i in range(0, len(tweets)):
            mention = re.findall(r'[@]\w+', tweets.iloc[i]['tweet'])
            mentions.append(mention)
        mentions = list(itertools.chain.from_iterable(mentions))
        # return string
        return mentions

    def choose_dataType(self):
        my_expander = st.sidebar.expander("Choose Data Source",expanded=True)
        with my_expander:
            options = ['Get Tweets From Twitter API','Get Tweets From CSV/EXCEL File']
            Data_Source = st.radio('', options)

            if Data_Source == "Get Tweets From Twitter API" :
                self.select_dataType = 'twitter'

            else:
                self.select_dataType = 'csv'
                self.usedetails = list()
        st.subheader(Data_Source)

    def get_table_download_link_csv(self, df):
        # csv = df.to_csv(index=False)
        csv = df.to_csv(index=False).encode()
        # b64 = base64.b64encode(csv.encode()).decode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
        return href

    def get_table_download_link_excel(self, df):
        towrite = io.BytesIO()
        downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
        towrite.seek(0)  # reset pointer
        b64 = base64.b64encode(towrite.read()).decode()  # some strings
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="myDataframe.xlsx">Download excel file</a>'
        return href

    def more_details(self):
        def twint_to_pd(columns):
            return twint.output.panda.Tweets_df[columns]
        def options():
            with st.form(key='my_form'):
                Search = st.text_input(label='Text To Search')
                Limit = st.text_input(label='Number of Tweets to Pull')

                if 'Username' in self.usedetails and 'Number of Likes' not in self.usedetails and 'Number of Replies' not in self.usedetails and 'Number of Retweets' not in self.usedetails:
                    Username = st.text_input(label='Write USER Name')

                if 'Period' in self.usedetails:
                    yesterday = datetime.date.today() + datetime.timedelta(days=-1)
                    today = datetime.date.today()
                    Since = st.date_input('Start date', yesterday)
                    until = st.date_input('End date', today)

                if 'Media' in self.usedetails:
                    select_media = st.radio('Display Tweets With Media or No', ('Only Tweets With Images or Videos', 'Only Tweets With Images', 'Only Tweets With Videos'))

                if 'Number of Likes' in self.usedetails:
                    if 'Username' in self.usedetails:
                        self.usedetails.remove('Username')
                    Min_likes = st.number_input('Minimum Number of Likes', min_value=0, max_value=1000000, value=200, step=1)

                if 'Number of Replies' in self.usedetails:
                    if 'Username' in self.usedetails:
                        self.usedetails.remove('Username')
                    Min_replies = st.number_input('Minimum Number of Replies', min_value=0, max_value=1000000, value=200, step=1)

                if 'Number of Retweets' in self.usedetails:
                    if 'Username' in self.usedetails:
                        self.usedetails.remove('Username')
                    Min_retweets = st.number_input('Minimum Number of Retweets', min_value=0, max_value=1000000, value=150, step=1)

                submit_button = st.form_submit_button(label='Search')

            if submit_button:
                x = 0
                c = twint.Config()  # Set up TWINT config
                c.Hide_output = True # Suppress Terminal Output when running search ...
                c.Search = Search  # Fill in the query that you want to search
                # ===================> Custom output format
                c.Limit = Limit  # Number of tweets to pull
                c.Pandas = True  # Storing objects in Pandas dataframes

                if 'Username' in self.usedetails:
                    c.Username = Username # choose username

                if 'Period' in self.usedetails:
                    c.Since = str(Since) # Filter tweets from this date
                    c.until= str(until) # Filter tweets upto this date
                    if Since > until:
                        x=1
                        st.error('Error: End date must fall after start date.')

                if 'Media' in self.usedetails:
                    if select_media =='Only Tweets With Images or Videos':
                        c.Media = True # Display tweets with only images or videos
                    elif select_media =='Only Tweets With Images':
                        c.Images= True # Display only tweets with images
                    else:
                        c.Vidoes = True  # Display only tweets with videos

                if 'Number of Likes' in self.usedetails:
                    c.Min_likes = int(Min_likes) # Filter tweets by minimum number of likes
                if 'Number of Replies' in self.usedetails:
                    c.Min_replies = Min_replies # Filter tweets by minimum number of retweets
                if 'Number of Retweets' in self.usedetails:
                    c.Min_retweets = Min_retweets # Filter tweets by minimum number of replies

                if x==0:
                    st.text('Loading data...')
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    twint.run.Search(c)
                    st.success('Loading data... done!')
                    self.tweet_df = twint_to_pd(['date', 'username', 'tweet', 'language', 'hashtags', 'nlikes', 'nreplies', 'nretweets', 'link', 'urls'])
                    st.write(self.tweet_df)
                    # save
                    tweet_df_CSV = twint_to_pd(['id', 'conversation_id', 'created_at', 'date', 'timezone', 'place',
                                                'tweet', 'language', 'hashtags', 'cashtags', 'user_id', 'user_id_str',
                                                'username', 'name', 'day', 'hour', 'link', 'urls', 'photos', 'video',
                                                'thumbnail', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url',
                                                'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
                                                'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
                                                'trans_dest'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(self.get_table_download_link_csv(tweet_df_CSV), unsafe_allow_html=True)
                    with col2:
                        pass
                    with col3:
                        st.markdown(self.get_table_download_link_excel(tweet_df_CSV), unsafe_allow_html=True)

        options()

    def select_details(self):
        if self.select_dataType == 'twitter':
            try:
                details = ['Username','Period','Media','Number of Likes','Number of Replies','Number of Retweets']
                label='Select fields'
                self.usedetails = st.multiselect(label=label, default='Period',options=details)
            except:
                st.write('Field Select Error')


    def edit_data(self):
        def options():
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button('Refresh', key='clear_cache1'):
                    try:
                        if self.df is not None:
                            del self.df
                    except:
                        pass
            with col2:
                pass
            with col3:
                self.select_col = st.checkbox('Select Specific Columns', value=False, key='2')

        def expander(key):
            options()

        my_expander = st.expander("Edit Read CSV", expanded=False)
        with my_expander:
            clicked = expander("filter")


    def select_cols(self):
        self.df_edit = self.tweet_df
        if self.df_edit is not None:
            try:
                df_columns = list(self.df_edit.columns)
                label='Select Columns'
                self.usecols = st.multiselect(label=label, default=df_columns,options=df_columns)
                self.df_edit = self.df_edit[self.usecols]
            except:
                st.write('Column Select Error')


    def parse_cols(self, df):
        df.columns = map(str.lower, df.columns)
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        return df.columns

#====================================================> Analyse de sentiments
    # Create a function to get the subjectivity
    def getSubjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    # Create a function to get the polarity
    def getPolarity(self, text):
        return TextBlob(text).sentiment.polarity

    #Create a function to compute the negative, neutral and positive analysis
    def getAnalysis(self, score):
      if score < 0:
        return 'Negative'
      elif score == 0:
        return 'Neutral'
      else:
        return 'Positive'
