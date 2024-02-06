import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Directly set the CSV filename here
CSV_FILENAME = 'testingthis.csv'  # Replace with the actual path to your CSV file

# Function to preprocess the header of the 4th column to remove '\r'
def preprocess_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        headers = file.readline().strip().split(',')  # Read the first line to get headers
        
    # Remove '\r' characters from the 4th header (index 3 since indexing starts from 0)
    if len(headers) >= 4:
        headers[3] = headers[3].replace('\r', '')

    return headers

if __name__ == "__main__":

    # Preprocess the headers
    headers = preprocess_header(CSV_FILENAME)

    # Read the CSV file with the corrected headers, skipping the original header row
    try:
        csv = pd.read_csv(CSV_FILENAME, lineterminator='\n', header=None, skiprows=1, names=headers)
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

    # Drop submission-related columns
    csv = csv.drop(columns=[
        'submission_name',
        'submission_text',
        '\r',   # Windows may append \r and it becomes considered
                # as its own column. This prevents that
    ])

    # Drop all rows whose labels are not 0 or 1
    csv = pd.concat([
        csv[csv['label'] == '0'], 
        csv[csv['label'] == '1'],
    ])

    # Drop all columns with no header
    # Prevents errors from having other unnecessary data in other columns
    # It selects the values of columns whose header does not
    # begin with 'Unnamed'
    csv = csv.loc[:, ~csv.columns.str.contains('^Unnamed')]

    # Extract X and y from the csv, this allows the data to be
    # undersampled
    X = csv.iloc[:, 0]
    y = csv.iloc[:, 1]

    # Reshape X into a 2D array to be compatible with the
    # undersampler
    X = X.values.reshape(-1, 1)

    # Initialize the random undersampler
    sampler = RandomUnderSampler(random_state=42)

    # Undersample the data
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    except ValueError:
        print("ERROR: Insufficient data")
        exit()

    # Flatten X again after resampling so it returns to
    # a 1D list
    X_resampled = X_resampled.flatten()

    # Make a new dataframe with the resampled data
    final_csv = pd.DataFrame(
        list(zip(X_resampled, y_resampled)),
        columns=['text', 'label']
    )

    # Save the filtered and resampled CSV
    final_csv.to_csv(
        CSV_FILENAME.replace('.csv', '-final.csv'),
        index=False,    # Prevent index from being saved as a column
    )

    print("Finished")
