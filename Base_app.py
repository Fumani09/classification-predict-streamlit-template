############ 1. IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation
import streamlit as st
import seaborn as sns
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pandas import json_normalize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolours
from wordcloud import WordCloud
import re
import joblib
import os


# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(page_title="BM3 Analytics", page_icon="💡",
                   initial_sidebar_state=st.session_state.sidebar_state, layout="centered")
##### SETTING UP ALL THE FUNCTIONS FOR THE VISUALIZATIONS AND EPLORATORY ANALYSIS ############

# Vectorizer
# news_vectorizer = open("resources/tfidfvect.pkl", "rb")
news_vectorizer = open("resources/TfidfVectorizer.pkl", "rb")

# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# loading the data


@st.cache_data
def load_csv_data():
    data_df = pd.read_csv("resources/streamlit.csv")
    return data_df


df = load_csv_data()

df['sentiment'] = df['sentiment'].map(
    {2: 'news', 1: 'pro', 0: 'neutral', -1: 'anti'})


@st.cache_data
def load_csv_data():
    data_data = pd.read_csv("resources/for_wordcloud.csv")
    return data_data


data = load_csv_data()
data['sentiment'] = data['sentiment'].map(
    {2: 'news', 1: 'pro', 0: 'neutral', -1: 'anti'})

# total number of tweets


def total_tweets(data):
    total_tweets = data.shape[0]
    return total_tweets
# total number of retweets


def total_retweets(data):
    retweeted_tweets = data[data['message']. str.startswith('RT')]
    num_retweets = len(retweeted_tweets)
    return num_retweets
# most cirulated usernames


def get_top_ten_names(data):
    retweeted_names = data[data['message'].str.startswith(
        'RT')]['message'].str.extract(r'RT @(\w+):')
    name_counts = retweeted_names[0].value_counts()
    top_ten_names = name_counts.head(10)
    return top_ten_names
# Function to remove retweets from text data


def remove_retweet(text):
    retweet_pattern = r'RT @\w+|@\w+'
    cleaned_text = re.sub(retweet_pattern, '', text)
    return cleaned_text


# people with the most retweets

######## PLOTS ###################

color_palette = ['#4CAF50']


def plot_retweeted_vs_not_retweeted(data):
    # Data
    retweeted_tweets = data[data['message'].str.startswith('RT')]
    num_retweets = len(retweeted_tweets)
    non_retweets = len(data) - num_retweets
    # Bar plot
    fig, ax = plt.subplots(facecolor='white')
    ax.bar(['Retweeted', 'Not Retweeted'], [
           num_retweets, non_retweets], color=color_palette)
    ax.set(xlabel='Tweet Type', ylabel='Number of Tweets',
           title='Number of Retweeted vs. Not Retweeted Tweets')
    ax.patch.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    plt.xticks(rotation=0)
    st.pyplot(fig)


# comparison of usernames most circulates the most top ten

def plot_top_ten_names(top_ten_names):
    # Create the bar chart
    fig, ax = plt.subplots(facecolor='white')
    ax.bar(top_ten_names.index, top_ten_names.values, color=color_palette)
    ax.set(xlabel='Usernames', ylabel='Tweet Counts',
           title='Top Ten Retweeted Usernames')
    ax.patch.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', rotation=45)
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    plt.xticks(rotation=90)
    # Display the chart using Streamlit
    st.pyplot(fig)

# comparison of people with the most retweets for the different sentiment scores


# Number of retweets per sentiment score
def plot_sentiment_scores(data):
    sentiment_counts = data['sentiment'].value_counts()
    sentiment_scores = sentiment_counts.index
    num_tweets = sentiment_counts.values

    # Bar plot
    fig, ax = plt.subplots(facecolor='white')
    ax.bar(sentiment_scores, num_tweets, color=color_palette)
    ax.set(xlabel='Sentiment Score', ylabel='Number of Tweets',
           title='Sentiment Scores per Number of Tweets')
    ax.patch.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    plt.xticks(rotation=0)
    st.pyplot(plt.gcf())


def plot_tweet_length(data):
    data['message'] = data['message'].apply(remove_retweet)
    data['tweet_length'] = data['message'].apply(lambda x: len(x.split()))

    # Box plot
    fig, ax = plt.subplots(facecolor='white')
    ax.boxplot([data[data['sentiment'] == sentiment]['tweet_length'] for sentiment in data['sentiment'].unique()], labels=data['sentiment'].unique(),
               patch_artist=True,
               boxprops=dict(facecolor='#4CAF50'),
               medianprops=dict(color='#000033'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    ax.set(xlabel='Sentiment Score', ylabel='Tweet Length',
           title='Tweet Length Distribution by Sentiment Score')
    ax.patch.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    st.pyplot(plt.gcf())


# number of hyperlinks per sentiment score from the data
def plot_hyperlink_bar_chart(df):
    # Define the pattern to find hyperlinks
    pattern = r'(https?://\S+)'

    # Find hyperlinks in each message
    df['hyperlinks'] = df['message'].apply(lambda x: re.findall(pattern, x))

    # Count the number of hyperlinks per sentiment score
    hyperlink_counts = df.explode('hyperlinks').groupby(
        'sentiment')['hyperlinks'].count()

    # Create the bar chart
    fig, ax = plt.subplots(facecolor='white')
    ax.bar(hyperlink_counts.index, hyperlink_counts.values, color=color_palette)
    ax.set(xlabel='Sentiment Score', ylabel='Number of Hyperlinks',
           title='Number of Hyperlink Patterns Based on Sentiment Score')
    ax.patch.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black', rotation=45)
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    plt.xticks(rotation=0)
    st.pyplot(fig)


# the most common words used for each of the sentiments


def create_wordcloud(data):
    all_messages = ' '.join(data['message'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, max_words=100,
                          background_color='white', prefer_horizontal=0.7).generate(all_messages)
    fig, ax = plt.subplots(facecolor='white')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############


############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.

c1, c2 = st.columns([1, 2])
with c1:

    st.image(
        "app_resources/images/logo_bm3_1.png",
        width=150,
    )
with c2:
    st.title("BM3 Analytics 💡")
# We need to set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

# setting the sidear to always be open


############ TABBED NAVIGATION ############

# First, we're going to create a tabbed navigation for the app via st.tabs()
# tabInfo displays info about the app.
# tabMain displays the main app.
# tabvis displays the visualizations of the app
# tabmod displays the models for the app


##############  SIDEBAR #####################
st.sidebar.subheader("Climate change tweet classification")
options = ["Logistic Regression Prediction",
           "Support Vector Prediction", "Multinomial NB Prediction"]
selection = st.sidebar.selectbox("Choose Option", options)

############# MAIN TAB CONTENTS ##############

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Twitter Sentiment Analysis')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))

MainTab, ModTab, InfoTab = st.tabs(["Main", "Model", "Info"])

with MainTab:
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2, 0.2, 1, 0.1))

    row1_spacer1, row1_1 = st.columns((0.1, 3.2))

    with row1_1:
        st.markdown(
            "Hey there!👋")
        st.markdown(" Get ready for an exhilarating data-driven adventure with our groundbreaking Twitter Sentiment Analysis project. We are delving into people's feelings about climate change using Twitter data. In today's sustainability-focused era, understanding public sentiments is crucial for individuals and businesses alike. With the power of Machine Learning, we're set to categorize views on climate change like never before. Join us on our Streamlit dashboard and let the data-driven exploration begin!🥳🥳🥳 ")

    st.header("Explore the data")

    ### SENTIMENT RANGE ###
    st.markdown("**First select the data range you want to analyze:** 👇")
    sentiment_options = df['sentiment'].unique().tolist()
    selected_sentiments = st.multiselect(
        'Select Sentiment', options=sentiment_options, default=sentiment_options)

    # Filter the dataset based on the selected sentiments
    filtered_df = df[df['sentiment'].isin(selected_sentiments)]
    filtered_df = filtered_df.reset_index(drop=True)
    # Initialize a boolean flag to track the visibility of the DataFrame
    show_data = False

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("Number of Tweets")
        total_tweets = filtered_df.shape[0]
        st.header(total_tweets)
    with col2:
        st.markdown("Number of retweets")
        retweeted_tweets = filtered_df[filtered_df['message']. str.startswith(
            'RT')]
        total_retweets = retweeted_tweets.shape[0]
        st.header(total_retweets)
    with col3:
        st.markdown("Unretweeted Tweets")
        retweeted_tweets = filtered_df[filtered_df['message']. str.startswith(
            'RT')]
        unretweeted_tweets = total_tweets - total_retweets
        st.header(unretweeted_tweets)

    # Create the dropdown
    with st.expander("Click here to first view the raw data"):
        st.dataframe(filtered_df)

    column_1, column_2 = st.columns([2, 3])

    with column_1:
        st.image(
            "app_resources/images/logo_bm3_2.png",
            width=60,
        )
        st.header("Retweeted vs Not Retweets")
        st.header("")
        st.markdown("This graph shows the relationship between the retweets an the tweets which are not retweeted. It is a strong indicator on whether opinions on climate are based off influence or independent thoughts")
    with column_2:
        plot_retweeted_vs_not_retweeted(filtered_df)

    st.header(" ")

    column_3, column_4 = st.columns([2, 3])

    with column_3:
        st.image(
            "app_resources/images/logo_bm3_2.png",
            width=60,
        )
        st.header("Retweets")
        st.markdown(
            "This bar chart shows how many tweets are present in the data for each of the sentiment scores")
        st.markdown(
            "It gives insights on the inclination and opinions of the tweet owners")

    with column_4:
        plot_sentiment_scores(filtered_df)

    st.header(" ")

    column_5, column_6 = st.columns([2, 3.5])

    with column_5:
        st.image(
            "app_resources/images/logo_bm3_2.png",
            width=60,
        )
        st.header("Top Ten Retweeted People")
        st.markdown("This graph shows how often the top ten retweeted people tweet and could be an indicator of which tweeters are the most active on topics pertinent to climate change.")

    with column_6:
        top_ten_names = get_top_ten_names(filtered_df)
        plot_top_ten_names(top_ten_names)

    st.header(" ")

    column_7, column_8 = st.columns([2, 3.5])

    with column_7:
        st.image(
            "app_resources/images/logo_bm3_2.png",
            width=60,
        )
        st.header(" Average Length of Tweets")
        st.markdown("This graph shows the average length of tweets with particular sentiment scores. The length of the tweet could bear information on how much information the tweeters might have on the topic of climate change or whether retweet long tweets of shorter tweets")

    with column_8:
        plot_tweet_length(filtered_df)

    st.header(" ")

    column_9, column_10 = st.columns([2, 3.5])

    with column_9:
        st.image(
            "app_resources/images/logo_bm3_2.png",
            width=60,
        )
        st.header("Hyperlinks Tweeted per Sentiment Score")
        st.markdown("This graph shows how many hyperlinks are tweeted per sentiment score. It is a string indicator of whether the tweets might contain external factual data or data on events that are relevant to climate change.")

    with column_10:
        plot_hyperlink_bar_chart(filtered_df)

    st.markdown("**First select the data range you want to analyze:** 👇")
    sentiment_options = data['sentiment'].unique().tolist()
    selected_sentiment = st.selectbox(
        'Select Sentiment', options=sentiment_options)
    st.header(" ")
    # Filter the dataset based on the selected sentiment
    filtered = data[data['sentiment'] == selected_sentiment]
    filtered = filtered.reset_index(drop=True)

    # Initialize a boolean flag to track the visibility of the DataFrame
    show_data = False
    st.markdown(" The wordcloud below shows the most common phrases used in tweets of various sentiment scores. These words give insights on the most commonly circulated hashtags and what phrases are likely to garner more views and retweets.")
    create_wordcloud(filtered)


########################################### INFORMATION ABOUT COOL TEAM BM4 ##########################################################################################
with InfoTab:
    image_width = 200

    st.subheader("About our company")
    st.markdown(
        "We are a leading data science company dedicated to helping businesses unlock the power of data to drive growth, innovation, and success. With our expertise in advanced analytics, machine learning, and artificial intelligence, we provide actionable insights and data-driven solutions that empower organizations to make informed decisions and achieve their goals."
    )
    st.header("")
    st.subheader("Our Expertise")
    st.markdown(
        """With a team of highly skilled data scientists, machine learning engineers, and domain experts, we have the knowledge and experience to tackle complex data challenges across various industries. From predictive modeling and data visualization to natural language processing and recommendation systems, we specialize in a wide range of data science techniques and technologies."""
    )
    st.header("")
    st.subheader("The Team")

    column_11, column_12 = st.columns([2, 2])
    with column_11:
        st.markdown("Fumani Thibela")
        st.image(
            "app_resources/images/Fumani.jpeg", width=image_width
        )
    with column_12:
        st.markdown("Mantsali Sekoli")
        st.image("app_resources/images/Mantsali.jpeg", width=image_width)

    column_11, column_12 = st.columns([2, 2])
    with column_11:
        st.markdown("Colette Muiruri")
        st.image("app_resources/images/Colette.jpeg", width=image_width)
    with column_12:
        st.markdown("Tercius Mapholo")
        st.image("app_resources/images/Tercius.jpg", width=image_width)


######################## TESTING OUT OUR MODEL AND APIS ###########################################################################
def make_predict(url):
    # Transforming user input with vectorizer
    vect_text = tweet_cv.transform([tweet_text]).toarray()
    # Load your .pkl file with the model of your choice + make predictions
    # Try loading in multiple models to give the user a choice
    predictor = joblib.load(
        open(os.path.join(url), "rb"))
    lr_prediction = predictor.predict(vect_text)

    match lr_prediction[0]:
        case 2:
            st.success(
                "Text Categorized as: {} - the tweet links to factual news about climate change".format(lr_prediction[0]))

        case 1:
            st.success(
                "Text Categorized as: {} - the tweet supports the belief of man-made climate change".format(lr_prediction[0]))

        case 0:
            st.success(
                "Text Categorized as: {} - the tweet neither supports nor refutes the belief of man-made climate change".format(lr_prediction[0]))

        case -1:
            st.success(
                "Text Categorized as: {} - the tweet does not believe in man-made climate change Variable definitions".format(lr_prediction[0]))

    # ff
    st.caption("")
    st.markdown("### Check the results!")
    st.info("For Further Information on Categories:  \n"
            "•	2 News: the tweet links to factual news about climate change  \n"
            "•	1 Pro: the tweet supports the belief of man-made climate change  \n"
            "•	0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change  \n"
            "•	-1 Anti: the tweet does not believe in man-made climate change Variable definitions")
    st.caption("")

    # st.write(df)


with ModTab:
    # st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.
    st.subheader("Climate change tweet classification")
    st.write("")
    st.markdown(
        """

    Classify keyphrases on the fly with this mighty twitter analysis app. No training needed!
   

    """
    )

    # Creating a text box for user input
    tweet_text = st.text_area("Enter Text", "Type Here")
    # Logistic Regression

    # Building out the predication page
    if selection == "Logistic Regression Prediction":
        if st.button("Submit"):
            url = "pickles/lr_model.pkl"
            make_predict(url)

    # Support Vector Machine
    # Building out the predication page
    if selection == "Support Vector Prediction":
        if st.button("Submit"):
            url = "pickles/svc_model.pkl"
            make_predict(url)

    # Naives Bayes
    # Building out the predication page
    if selection == "Multinomial NB Prediction":
        if st.button("Submit"):
            url = "pickles/mnb_model.pkl"
            make_predict(url)
