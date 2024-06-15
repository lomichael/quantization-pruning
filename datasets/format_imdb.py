import pandas as pd

# Load the original dataset
df = pd.read_csv('IMDB Dataset.csv')

# Select only the 'review' column
formatted_df = df[['review']]

# Save the formatted dataset
formatted_df.to_csv('data.csv', index=False)
