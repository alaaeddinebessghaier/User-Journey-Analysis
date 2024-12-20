import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('C:\\Users\\bessghaier\\365 project\\project-files-user-journey-analysis-in-python\\user_journey_raw.csv')
print(df.head())
print(df.describe(include='all'))
print(df.info())
""" print(df.isnull().values.sum()) 
print(df['user_id'].duplicated().sum())
print(df['session_id'].duplicated().sum()) """

### top user_journey used by user
journey_counts = df['user_journey'].value_counts()
most_duplicated_journey = journey_counts.sort_values(ascending=False,axis=0)
""" print(most_duplicated_journey.head(10))"""


### most subscription_type by user 
user_session = df.groupby('subscription_type')['user_id'].count()
""" print(user_session) 
 """
### user have more session 
user = df['user_id'].value_counts()
most_duplicated_user = user.sort_values(ascending=False)
""" print(most_duplicated_user)
 """

### remove the duplicated pages
def remove_page_duplicates(df,column='user_journey'):
    cleaned_journeys = []
    for index,row in df.iterrows():
        journey = row[column]
        pages = journey.split('-')
        clean_pages=[pages[0]]
        for i in range(1,len(pages)):
            if pages[i] != pages[i-1]:
                clean_pages.append(pages[i])
        clean_pages = '-'.join(clean_pages)
        cleaned_journeys.append(clean_pages)
    df_cleaned = df.copy()
    df_cleaned[column] = cleaned_journeys
    return df_cleaned
result = remove_page_duplicates(df)
print(result)


### group the sessions of the user 
def group_by(df,sessions,count_from,group_column='user_id',target_column='user_journey'):
    data_sorted = df.sort_values(by=['user_id', 'session_id'], ascending=True)
    journey = data_sorted.groupby(group_column)[target_column].apply(list)

    if sessions == 'all':
        return journey
    else:
        if count_from == 'first':
            journey= journey.apply(lambda x: x[:sessions])
        elif count_from == 'last':
            journey= journey.apply(lambda x: x[-sessions:])
        return journey.reset_index(name='grouped_journey')
    
sz = group_by(df, 3, 'last')
print(sz)
print(len(sz))
print(df['user_id'].nunique())

