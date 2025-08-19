
import pandas as pd
import numpy as np
from collections import Counter
from typing import List
import re 
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


def generate_features(df: pd.DataFrame, batch_size: int = 100):

    df = df.copy()

    df = df.dropna(subset=["text"])

    df = reply_threaded_behavior(df)

    df = sentiment_analysis(df)

    df = time_information_features(df)

    # df = df.dropna()


    df.loc[:,"text"] = df["text"].apply(lambda x: len(x.split()))  

    df.loc[:,"num_urls"] = df["num_urls"].fillna(0).apply(lambda x: int(x))

    df.loc[:,"num_mentions"] = pd.to_numeric(df["num_mentions"], errors="coerce").fillna(0).apply(lambda x: int(x))
 
    df.loc[:,"num_hashtags"] = df["num_hashtags"].fillna(0).apply(lambda x: int(x))

    df.loc[:,"possibly_sensitive"] = pd.to_numeric(df["possibly_sensitive"], errors="coerce").fillna(0).apply(lambda x: int(x))

    df.loc[:,"favorite_count"] = df["favorite_count"].fillna(0).apply(lambda x: int(x))
 
    df.loc[:,"reply_count"] = df["reply_count"].fillna(0).apply(lambda x: int(x))
 
    df.loc[:,"retweet_count"] = df["retweet_count"].fillna(0).apply(lambda x: int(x))

    df.loc[:,"truncated"] = pd.to_numeric(df["truncated"], errors="coerce").fillna(0).apply(lambda x: int(x))


    df = metadata_features(df)
    df = df.sample(frac=1).reset_index(drop=True)
    df[['favorite_count', 'num_urls', 'reply_count', 'retweet_count','num_hashtags', 'possibly_sensitive', 
            'num_mentions', 'text', 'time_diff_created_crawled', 'timestamp_in_future', 'created_at_hour', 'time_of_day', 
            'is_reply', 'is_reply_to_user', 'is_reply_to_screen_name', 'hashtags_mentions_ratio', 'sentiment', 
            'hashtags_urls_ratio', 'mentions_urls_ratio', 'label']].to_csv('data/tweets/data.csv')
    print(df.head())

def time_information_features(df: pd.DataFrame):
    """
        Time Information Feature Generation
    """

    df.loc[:, 'created_at'] = pd.to_datetime(df['created_at'], format="mixed", errors='coerce',utc=True)
    df.loc[:,'crawled_at'] = pd.to_datetime(df['crawled_at'], format='mixed', errors='coerce', utc=True)
    
    df = df.dropna(subset=['created_at','crawled_at'])
   
    df['created_at'] = pd.to_datetime(df['created_at'],utc=True)
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], utc=True)

    df.loc[:,'timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce', utc=True)
    df.loc[:,'updated'] = pd.to_datetime(df['updated'], format='mixed', errors='coerce', utc=True)
    
    df.loc[:,'created_at'] = df['created_at'].dt.tz_localize(None)
    df.loc[:,'crawled_at'] = df['crawled_at'].dt.tz_localize(None)
    
    df.loc[:,'time_diff_created_crawled'] = (df['crawled_at'] - df['created_at']).dt.days

  
    df.loc[:,'timestamp_in_future'] = df['timestamp'] > pd.to_datetime('now', utc=True)

    
    df.loc[:,'created_at_hour'] = df['created_at'].dt.hour
    df.loc[:,'time_of_day'] = pd.cut(df['created_at_hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'], right=False)

    return df


def reply_threaded_behavior(df):
    df.loc[:,'is_reply'] = df['in_reply_to_status_id'].notna().astype(int)
    df.loc[:,'is_reply_to_user'] = df['in_reply_to_user_id'].notna().astype(int)
    df.loc[:,'is_reply_to_screen_name'] = df['in_reply_to_screen_name'].notna().astype(int)

    return df


def sentiment_analysis(df):
    df.loc[:,'sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df


def metadata_features(df):
    # df['num_hashtags'] = df['text'].apply(lambda x: len(re.findall(r'#\w+', x)))
    # df['num_mentions'] = df['text'].apply(lambda x: len(re.findall(r'@\w+', x)))
    # df['num_urls'] = df['text'].apply(lambda x: len(re.findall(r'http[s]?://\S+', x)))
    
    df.loc[:,'hashtags_mentions_ratio'] = df['num_hashtags'] / (df['num_mentions'] + 1) 
    df.loc[:,'hashtags_urls_ratio'] = df['num_hashtags'] / (df['num_urls'] + 1) 
    df.loc[:,'mentions_urls_ratio'] = df['num_mentions'] / (df['num_urls'] + 1)  
    
    return df

def chunk_users_data(df: pd.DataFrame, n: int = 50) -> List[pd.DataFrame]:
    """
    Chunk data into smaller batches based on user_id
    """
    return [df.iloc[i:i + n] for i in range(0, len(df), n)]

if __name__ == "__main__":

    # spam bot data
    with open("data/tweets/social_spambots_v2.csv", "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    with open("data/tweets/temp.csv", "w", encoding="utf-8", errors="ignore") as f:
        f.write(data)
    timeline_data = pd.read_csv("data/tweets/temp.csv", encoding="utf-8", on_bad_lines="skip", dtype=str)
    timeline_data.loc[:,'label'] = 0

    # genuine user data
    chunk_size = 500000
    for chunk in pd.read_csv("data/tweets/genuine_users_.csv", encoding="utf-8", chunksize=chunk_size, on_bad_lines="skip", dtype=str):
        chunk.loc[:, 'label'] = 1
        timeline_data = pd.concat([timeline_data, chunk])
        break
  
    # Index(['id', 'text', 'source', 'user_id', 'truncated', 'in_reply_to_status_id',
    #     'in_reply_to_user_id', 'in_reply_to_screen_name', 'retweeted_status_id',
    #     'geo', 'place', 'contributors', 'retweet_count', 'reply_count',
    #     'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive',
    #     'num_hashtags', 'num_urls', 'num_mentions', 'created_at', 'timestamp',
    #     'crawled_at', 'updated'],
    #     dtype='object')
    # """

    generate_features(timeline_data)

   

