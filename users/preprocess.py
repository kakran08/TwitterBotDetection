from datetime import datetime
import pandas as pd
from typing import List

def preprocess_account(df: pd.DataFrame):
    """
    Preprocesses the Twitter data frame and creates features.
    """
    df = df.copy()
    df.loc[:, "geo_enabled"] = df["geo_enabled"].isna().apply(lambda x: int(x))
    df.loc[:,"default_profile_image"] = df["default_profile_image"].isna().apply(lambda x: int(x))
    df.loc[:,"default_profile"] = df["default_profile"].isna().apply(lambda x: int(x))
    df.loc[:,"statuses_count"] = pd.to_numeric(df["statuses_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"protected"] = pd.to_numeric(df["protected"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"verified"] = pd.to_numeric(df["verified"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"description"] = df["description"].fillna('').apply(lambda x: len(x))
    df.loc[:,"following"] = pd.to_numeric(df["following"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"followers_count"] = pd.to_numeric(df["followers_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"friends_count"] = pd.to_numeric(df["friends_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"favourites_count"] = pd.to_numeric(df["favourites_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"statuses_count"] = pd.to_numeric(df["statuses_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"listed_count"] = pd.to_numeric(df["listed_count"], errors="coerce").fillna(0).apply(lambda x: int(x))
    df.loc[:,"url_count"] = df["url"].isna().apply(lambda x: int(x))
    df.loc[:, 'location'] = df["location"].isna().apply(lambda x: int(x))
    df.loc[:, 'notifications'] = df["notifications"].isna().apply(lambda x: int(x))
    df.loc[:, 'updated'] = df["updated"].isna().apply(lambda x: int(x))
    
    df.loc[:, 'created_at'] = pd.to_datetime(df['created_at'], format="mixed", errors='coerce',utc=True)
    df.loc[:,'crawled_at'] = pd.to_datetime(df['crawled_at'], format='mixed', errors='coerce', utc=True)
    
    # df = df.dropna(subset=['created_at','crawled_at'])
   
    df['created_at'] = pd.to_datetime(df['created_at'],utc=True)
    df['crawled_at'] = pd.to_datetime(df['crawled_at'], utc=True)

    df.loc[:,"time_diff"] = (df['crawled_at'] - df['created_at']).dt.days 

    df = df.sample(frac=1).reset_index(drop=True)

    df[["geo_enabled", "default_profile_image", "default_profile", "statuses_count", "protected", "verified", "description", "following", "followers_count", "friends_count", 
        "favourites_count", "favourites_count", "statuses_count", "listed_count", "url_count", "location", "notifications", "updated", "time_diff", "label"]].to_csv('data/users/data.csv')

    print(df.head())

if __name__ == "__main__":

    gu = pd.read_csv('data/users/genuine_users.csv')
    ssb1 = pd.read_csv('data/users/social_spambots_v1.csv')
    ssb2 = pd.read_csv('data/users/social_spambots_v2.csv')
    ssb3 = pd.read_csv('data/users/social_spambots_v3.csv')

    ssb1["test_set_2"] = 0
    ssb2["test_set_1"] = 0
    ssb3["test_set_1"] = 0
    ssb3["test_set_2"] = 0

    ssb1["label"] = 0
    ssb2["label"] = 0
    ssb3["label"] = 0
    gu["label"] = 1

    df = pd.concat([gu,ssb1,ssb2,ssb3], axis=0)
    preprocess_account(df)