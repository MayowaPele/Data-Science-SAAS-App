from django.shortcuts import render
from django.http import *
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from plotly.offline import plot
import plotly.graph_objects as go

# Create your views here.
class TwitterSentClass():
    def __init__(self):
        API_key = "HFpKY8Klhkcf07Tvt7P6OiQfi"
        API_secret = "thWfyz5Yi9hMgZxrwlQ8lz6YkjTIIoMyRvUY4neI4KKolQWnQQ"
        access_token = "1266307077481340928-ptm14eUWq9QYGlhOls72edevAX5FD2"
        access_secret = "IHp6dLGORUMJ8qEb1M47zMe2iBFKFQ0ErhUKwnfRBew3k"
        try:
            self.auth = OAuthHandler(API_key, API_secret)
            self.auth.set_access_token(access_token, access_secret)
            self.api = tweepy.API(self.auth)
            print('Authenticated')
        except:
            print('Sorry! Error in authentication')

    def cleaning_process(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
                               , " ", tweet).split())

    def get_sentiment(self, tweet):
        analysis = TextBlob(self.cleaning_process(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=1000):
        tweets = []
        try:
            fetched_tweets = self.api.search(q = query, count = count)
            print(fetched_tweets[0:2])
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_sentiment(tweet.text)
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                        tweets.append(parsed_tweet)
            return tweets
        
        except tweepy.TweepError as e:
            print("Error: "+ str(e))

def sentiment_main(request):
    return render(request, 'sentiment_analyser/base.html')

def prediction(request):
    if request.method == 'POST':
        api = TwitterSentClass()
        t = request.POST['link']
        tweets = api.get_tweets(query = t, count=100)

        pos_tweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        neg_tweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        neutral_tweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral']

        nos_of_positive = len(pos_tweets)
        nos_of_negative = len(neg_tweets)
        nos_of_neutral = len(neutral_tweets)

        def bar():
            data = go.Bar(
                x=['Positive','Negative', 'Neutral'],
                y=[nos_of_positive, nos_of_negative, nos_of_neutral]
            )

            layout = dict(
                title="Twitter Graph Sentiments",
                yaxis=dict(range=[0,100])
            )

            fig = go.Figure(data=[data], layout=layout)
            plot_div = plot(fig, output_type='div', include_plotlyjs=False)
            return plot_div

        context = {
                'plot': bar()
            }     

    return render (request, 'sentiment_analyser/base.html', context)
