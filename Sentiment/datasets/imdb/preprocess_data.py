# %%
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
# Load the dataset
df = pd.read_csv(
    'C:\\Users\\Yuval\\Desktop\\Training-an-LLM\\Sentiment\\datasets\\imdb\\rawdata.csv', encoding='utf-8')

# %%
# Transform the sentiment to the fastText format
df['sentiment'] = '__label__' + df['sentiment']

# Remove all double quotes from the review and combine with sentiment
df['review'] = df['sentiment'] + ' ' + df['review']

# Splitting the data
train, temp = train_test_split(
    df['review'], test_size=0.5, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)


def foo(df, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in df:
            file.write(line + '\n')


dataframes = {'train.txt': train, 'dev.txt': dev, 'test.txt': test}

for name, df in dataframes.items():
    foo(df, name)
