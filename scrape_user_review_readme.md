
# Metacritic User Review Scraper

This Python script allows you to scrape user reviews from Metacritic based on game slugs and platform slugs from a CSV file. It processes each row, fetches user review data using the Metacritic API, and saves the updated information into a new CSV file.

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.x installed on your machine
- Required Python packages installed (`requests`, `csv`)

You can install the required packages using pip:
```bash
pip install requests
```

## Setup

1. **Download or Clone the Repository:**
   - Download the script or clone this repository to your local machine.

2. **Prepare the Input CSV:**
   - The script expects a CSV file with at least the following columns:
     - `Slug`: The unique identifier for the game (used to get game reviews).
     - `Platform Slug`: The platform the game is on (e.g., `ps4`, `xbox`).
     
   Example of the CSV file:
   ```csv
   Slug, Platform Slug, Game Name
   the-last-of-us-2, ps4, The Last of Us 2
   god-of-war, ps5, God of War
   ```

3. **Obtain Metacritic API Key:**
   - Obtain your API key from Metacritic by registering or using the API if it's available to you.

4. **Modify the Script:**
   - Update the script with your input and output file paths and the API key.

```python
api_key = "your_api_key_here"
input_file = "path_to_your_input_file.csv"
output_file = "path_to_your_output_file.csv"
```

5. **Run the Script:**
   - Run the script by executing the following command in your terminal or command prompt:
   
```bash
python metacritic_review_scraper.py
```

   - The script will process each row in the input file, fetch the user reviews for each game/platform pair, and save the updated data in the output CSV file.

## Script Details

### Main Functions

1. **`fetch_api_data(api_key, base_url, game_slug, platform_slug)`**:
   - Fetches data from the Metacritic API for the given `game_slug` and `platform_slug`.
   - Returns a JSON object with the review data.

2. **`fetch_user_reviews(game_slug, platform_slug, api_key)`**:
   - Calls the API to fetch the reviews and processes the response to extract the necessary review information (e.g., user score, sentiment, review counts).

3. **`handle_null_values(row)`**:
   - Replaces null, empty, or invalid values in a row with `"N/A"` to ensure the output data is complete.

4. **`save_to_csv(data, filename)`**:
   - Saves the updated data into a new CSV file.

5. **`main()`**:
   - The entry point of the script that reads the input file, processes each row, fetches the review data, handles missing values, and saves the updated data to the output file.

### Error Handling

- **FileNotFoundError**: If the input file is not found, the script will print an error message.
- **General Errors**: Other errors will be caught and printed for debugging.

## Example Output CSV

The output CSV will contain the original game and platform information along with the fetched review data.

| Slug             | Platform Slug | Game Name       | User Score | User Review Count | User Positive Count | User Neutral Count | User Negative Count | User Sentiment | User Review URL |
|------------------|---------------|-----------------|------------|-------------------|---------------------|--------------------|---------------------|-----------------|-----------------|
| the-last-of-us-2 | ps4           | The Last of Us 2| 9.5        | 500               | 450                 | 30                 | 20                  | Positive        | https://example.com |
| god-of-war       | ps5           | God of War      | 8.7        | 350               | 300                 | 40                 | 10                  | Neutral         | https://example.com |

## Troubleshooting

- **Missing Data:** If a game or platform has no available data, the script will replace missing values with `"N/A"`.
- **API Errors:** If the API request fails, the script will skip that row and print an error message.

## License

This script is provided as-is. You can freely use, modify, and distribute it for personal or educational purposes.

---

For more information on how to use the Metacritic API, check their official documentation or refer to the API terms of service.
