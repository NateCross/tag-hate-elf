import csv
import os
import re

# Specify the input and output file paths (use absolute paths)
current_directory = os.getcwd()
input_file = os.path.join(current_directory, "datasets", "test.csv")
output_file = os.path.join(current_directory, "datasets", "output.csv")


# Function to read the CSV file and modify the first column
def process_csv(input_file, output_file):
    with open(input_file, "r", newline="", encoding="utf-8") as csv_in, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out, quoting=csv.QUOTE_NONE, escapechar=" ")

        # Write the header row
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            # Check if there are at least two columns in the row
            if len(row) >= 2:
                # Process the content in the first column
                content = row[0].replace(
                    "\n", " "
                )  # Replace newline characters with spaces
                content = re.sub(
                    r"\s+", " ", content
                )  # Replace multiple spaces with a single space
                content = content.strip()  # Remove leading and trailing spaces
                content = content.replace(",", " ")  # Remove commas
                row[0] = f'"{content}"'

            # Write the modified row to the output file
            writer.writerow(row)


if __name__ == "__main__":
    process_csv(input_file, output_file)
    print(
        f'CSV file "{input_file}" has been processed. \n Modified data is saved to "{output_file}".'
    )
