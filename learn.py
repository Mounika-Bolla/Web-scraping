import requests
import urllib.parse
from PIL import Image
import pytesseract
import os
import pandas as pd
import re
import time

# Function to capture a screenshot with retry mechanism
def capture_screenshot(url, width, height, output_path, retries=3):
    BASE = f'https://mini.s-shot.ru/{width}x{height}/JPEG/1280/Z100/?'
    encoded_url = urllib.parse.quote_plus(url)
    full_url = BASE + encoded_url
    
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(full_url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    for chunk in response:
                        file.write(chunk)
                print(f"Screenshot saved as {output_path}")
                return True
            else:
                print(f"Failed to capture screenshot for {url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error capturing screenshot for {url}: {str(e)}")
        
        attempt += 1
        print(f"Retrying... ({attempt}/{retries})")
        time.sleep(2)  # Delay between retries
    return False

# Function to crop table from the image
def crop_table_from_image(image_path, output_path, crop_box):
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(crop_box)
        cropped_img.save(output_path)
        print(f"Cropped table saved as {output_path}")
    except Exception as e:
        print(f"An error occurred while cropping the image {image_path}: {str(e)}")

# Function to extract text from image and print for debugging
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print(f"Extracted text from {image_path}:\n{text}\n")  # Debug: Print extracted text
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from image {image_path}: {str(e)}")
        return None

# Function to parse total volume and at close values from the extracted text
def parse_total_volume_and_at_close(text, month_code):
    print(f"Parsing text for month code {month_code}...")  # Debugging: Show parsing step
    lines = text.split('\n')
    for line in lines:
        print(f"Processing line: {line}")  # Debug: Show each line being processed
        if month_code in line:
            # Extract total volume and at close using regex or text parsing
            match = re.findall(r'\b(\d+)\b', line)
            if len(match) >= 2:
                total_volume = match[4]
                at_close = match[-2]
                return total_volume, at_close
    return None, None

# Create output directories
screenshots_dir = 'screenshots'
cropped_dir = 'cropped_images'
os.makedirs(screenshots_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# Sample data for Instruments and HTML Links
data = {
    'Instrument': [
        'DOW', 'ES', 'EUR', 'FDAX', 'FGBL', 'GOLD', 
        'WEST TEXAS CRUDE OIL', 'MICRO GOLD', 'US 30-YR BONDS', 
        'NASDAQ', 'MICRO NASDAQ', 'SOY BEANS', 'GBP', 'YEN'
    ],
    'HTML Link': [
        'https://www.cmegroup.com/markets/equities/dow-jones/e-mini-dow.volume.html',
        'https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.volume.html',
        'https://www.cmegroup.com/markets/fx/g10/euro-fx.volume.html',
        'NONE', 'NONE',
        'https://www.cmegroup.com/markets/metals/precious/gold.volume.html',
        'https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.volume.html',
        'https://www.cmegroup.com/markets/metals/precious/e-micro-gold.volume.html',
        'https://www.cmegroup.com/markets/interest-rates/us-treasury/30-year-us-treasury-bond.volume.html',
        'https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.volume.html',
        'https://www.cmegroup.com/markets/equities/nasdaq/micro-e-mini-nasdaq-100.volume.html',
        'https://www.cmegroup.com/markets/agriculture/oilseeds/soybean.volume.html',
        'https://www.cmegroup.com/markets/fx/g10/british-pound.volume.html'
    ],
    'Month Code': [
        'Z24', 'Z24', 'Z24', 'Z24', 'Z24', 'Z24',
        'X24', 'X24', 'Z24', 'Z24', 'Z24', 'Z24', 'Z24'
    ]
}

# DataFrame initialization
df = pd.DataFrame(data)

# Loop over the data and process each URL
for index, row in df.iterrows():
    instrument = row['Instrument']
    url = row['HTML Link']
    month_code = row['Month Code']
    
    if url != 'NONE':
        # Capture screenshot for each URL with retries
        screenshot_path = f'{screenshots_dir}/{instrument}_screenshot.jpg'
        success = capture_screenshot(url, 1280, 4000, screenshot_path)
        
        if success:
            # Crop the image
            crop_box = (20, 1350, 1280, 1800)  # Example crop box, adjust based on actual table position
            cropped_image_path = f'{cropped_dir}/{instrument}_cropped.png'
            crop_table_from_image(screenshot_path, cropped_image_path, crop_box)
            
            # Extract text from the cropped image
            extracted_text = extract_text_from_image(cropped_image_path)
            
            if extracted_text:
                # Parse total volume and at close based on the month code
                total_volume, at_close = parse_total_volume_and_at_close(extracted_text, month_code)
                
                # Add the extracted data to the DataFrame
                df.at[index, 'Total Volume'] = total_volume
                df.at[index, 'At Close'] = at_close
            else:
                df.at[index, 'Total Volume'] = 'N/A'
                df.at[index, 'At Close'] = 'N/A'
        else:
            df.at[index, 'Total Volume'] = 'Screenshot Failed'
            df.at[index, 'At Close'] = 'Screenshot Failed'
    else:
        df.at[index, 'Total Volume'] = 'NONE'
        df.at[index, 'At Close'] = 'NONE'

# Display the updated DataFrame
print(df)

# Optionally save the DataFrame
df.to_csv('instruments_with_total_volume_and_at_close.csv', index=False)
