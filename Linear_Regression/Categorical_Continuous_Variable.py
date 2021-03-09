import pandas as pd

df = pd.DataFrame(['0-10','20-30','30-40','20-30','50-60','70-80'], columns=['age'], index=[1,2,3,4,5,6])
print(df.shape)
print(df.size)
print(df)

df[['start', 'end']] = df['age'].str.split('-',expand = True)
print(df)

def mean(x):
    lst = x.split('-')
    print(lst)
    value = float(lst[0]) + float(lst[1])
    return value

df['mean_age'] = df['age'].apply(lambda x: mean(x))
print(df)

table_df = pd.crosstab(df['age'], df['mean_age'] ,margins=True)
print(table_df)
