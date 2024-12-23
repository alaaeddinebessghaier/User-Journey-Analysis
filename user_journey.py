import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


user_data = pd.read_csv('C:\\Users\\bessghaier\\365 project\\project-files-user-journey-analysis-in-python\\user_journey_raw.csv')
""" print(user_data.info)
print(user_data.describe())
print(user_data.dtypes) """


def remove_page_duplicates(data, target_column='user_journey'):
    df = data.copy()
    all_journey = list(df[target_column])
    all_journey = [journey.split("-") for journey in all_journey]
    for i in range(len(all_journey)):
        cleaned_journey = [all_journey[i][0]]
        for j in range(1,len(all_journey[i])):
            if all_journey[i][j-1] != all_journey[i][j]:
                cleaned_journey.append(all_journey[i][j])
        all_journey[i] = "-".join(cleaned_journey)
        df[target_column] = all_journey
    return df 



def group_by(data, group_column = 'user_id', target_column = 'user_journey', sessions = 'all', count_from = 'last'):
    
    

    
    df = pd.DataFrame(columns = data.columns)
    
    
    if sessions == "all":
        start = 0
        end = None
    
    elif sessions == "all_except_last":
        start = 0
        end = -1
    
    elif count_from == "last":
        start = - sessions
        end = None
    
    elif count_from == "first":
        start = 0
        end = sessions
    
    
    groups = set(data[group_column])
    
    for group_value in groups:
        
        group_mask = list(data[group_column] == group_value)
        group_table = data[group_mask] 
        
        user_journey = "-".join(list(group_table[target_column])[start:end])
        
        new_index = len(df)
        df.loc[new_index] = group_table.iloc[0].copy() 
        df.loc[new_index, target_column] = user_journey

    
    df.sort_values(by=[group_column], ignore_index = True, inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    
    
    return df

def remove_pages(data, pages=[], target_column='user_journey'):

    
    df = data.copy()  
    journes = list(df[target_column])  
    pages = set(pages)  
    if len(pages) == 0:
        return df 
    
    journes = [journey.split('-') for journey in journes]  
    journes = [[page for page in journey if page not in pages] for journey in journes]  
    journes = ["-".join(i) for i in journes]  
    
    df[target_column] = journes  
    return df  









clean_data = user_data.copy().drop("session_id", axis=1)
clean_data = group_by(clean_data)
print(type(clean_data))  # This should print <class 'pandas.core.frame.DataFrame'>

clean_data = remove_pages(clean_data, [])
print(type(clean_data))  # This should print <class 'pandas.core.frame.DataFrame'>

clean_data = remove_page_duplicates(clean_data)
print(clean_data.head())






