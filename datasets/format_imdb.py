import pandas as pd

# Load the original dataset
print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv')
print("Dataset loaded. Checking contents...")
print("First few rows of the dataset:")
print(df.head())

if not df.empty and 'review' in df.columns:
	print("Review column found. Processing data...")

	# Select only the 'review' column
	reviews = df['review']
	print("First few reviews:")
	print(reviews.head())

	# Create a DataFrame with only the 'review' column
	formatted_df = pd.DataFrame(reviews)
	formatted_df.columns = ['text']
	print("First few rows of the formatted DataFrame:")
	print(formatted_df.head())

	# Save the formatted dataset
	formatted_df.to_csv('data.csv', index=False)
	print("Data processing complete. Saved to data.csv.")

else:
	print("Error: DataFrame is empty or 'review' column not found")
