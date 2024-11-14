import requests
import urllib.parse
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imutils
from pytesseract import Output
from sklearn.cluster import AgglomerativeClustering
from tabulate import tabulate
import cv2


def capture_screenshot(url, width, height, output_path):
    """
    Function to capture a screenshot of a webpage using mini.s-shot.ru.
    """
    BASE = f'https://mini.s-shot.ru/{width}x{height}/JPEG/1280/Z100/?'
    encoded_url = urllib.parse.quote_plus(url)
    full_url = BASE + encoded_url
    
    response = requests.get(full_url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response:
                file.write(chunk)
        print(f"Screenshot saved as {output_path}")
    else:
        print(f"Failed to capture screenshot. Status code: {response.status_code}")

def crop_table_from_image(image_path, output_path, crop_box):
    """
    Function to crop the table region from an image.
    """
    # Load the image
    img = Image.open(image_path)
    
    # Crop the image using the coordinates (left, upper, right, lower)
    cropped_img = img.crop(crop_box)
    
    # Save the cropped image
    cropped_img.save(output_path)
    
    print(f"Cropped table saved as {output_path}")

# Step 1: Capture the screenshot
url = 'https://www.cmegroup.com/markets/equities/dow-jones/e-mini-dow.volume.html'
screenshot_path = 'cme_screenshot_large.jpg'

# Screenshot size (you can adjust these dimensions if necessary)
capture_screenshot(url, 1280, 4000, screenshot_path)

# Step 2: Crop the table from the screenshot
crop_box = (20, 1350, 1280, 1800)  # Example coordinates to crop the table from the screenshot
output_cropped_table = 'cropped_table_image.png'

# Crop the table region and save
crop_table_from_image(screenshot_path, output_cropped_table, crop_box)
image_file = output_cropped_table
args = {
    "image": image_file,
    "output": "results.csv",
    "min_conf": 0,
    "dist_thresh": 25.0,
    "min_size": 2, #2
}


image = cv2.imread(image_file) #PNG image file
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize a rectangular kernel that is ~5x wider than it is tall,
# then smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light background
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (500, 100))
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# compute the Scharr gradient of the blackhat image and scale the result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")

# apply a closing operation using the rectangular kernel to close gaps in between characters, 
#apply Otsu's thresholding method, and finally a dilation operation to enlarge foreground regions
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
thresh = cv2.threshold(grad, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, None, iterations=3)


# find contours in the thresholded image and grab the largest one,
# which we will assume is the stats table
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
tableCnt = max(cnts, key=cv2.contourArea)
# compute the bounding box coordinates of the stats table and extract
# the table from the input image
(x, y, w, h) = cv2.boundingRect(tableCnt)
table = image[y:y + h, x:x + w]


# Set the PSM mode to detect sparse text, and then localize text in the table
options = "--psm 6" 
results = pytesseract.image_to_data(
    cv2.cvtColor(table, cv2.COLOR_BGR2RGB),
    config=options,
    output_type=Output.DICT)

# Initialize a list to store the (x, y)-coordinates of the detected text along with the OCR'd text itself
coords = []
ocrText = []

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(float(results["conf"][i]))

    # filter out weak confidence text localizations
    if conf > args["min_conf"]:
        # update our text bounding box coordinates and OCR'd text,
        # respectively
        coords.append((x, y, w, h))
        ocrText.append(text)

# Extract all x-coordinates from the text bounding boxes, setting the y-coordinate value to zero
xCoords = [(c[0], 0) for c in coords]

# Apply hierarchical agglomerative clustering to the coordinates
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="manhattan",
    linkage="complete",
    distance_threshold=args["dist_thresh"])
clustering.fit(xCoords)

# Initialize our list of sorted clusters
sortedClusters = []

# loop over all clusters
for l in np.unique(clustering.labels_):
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]

    # verify that the cluster is sufficiently large
    if len(idxs) > args["min_size"]:
        # compute the average x-coordinate value of the cluster and
        # update our clusters list with the current label and the
        # average x-coordinate
        avg = np.average([coords[i][0] for i in idxs])
        sortedClusters.append((l, avg))

# sort the clusters by their average x-coordinate and initialize our
# data frame
sortedClusters.sort(key=lambda x: x[1])
df = pd.DataFrame()

# loop over the clusters again, this time in sorted order
for (l, _) in sortedClusters:
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]

    # extract the y-coordinates from the elements in the current
    # cluster, then sort them from top-to-bottom
    yCoords = [coords[i][1] for i in idxs]
    sortedIdxs = idxs[np.argsort(yCoords)]

    # generate a random color for the cluster
    color = np.random.randint(0, 255, size=(3,), dtype="int")
    color = [int(c) for c in color]

    # loop over the sorted indexes
    for i in sortedIdxs:
        # extract the text bounding box coordinates and draw the
        # bounding box surrounding the current element
        (x, y, w, h) = coords[i]
        cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)

    # extract the OCR'd text for the current column, then construct
    # a data frame for the data where the first entry in our column
    # serves as the header
    cols = [ocrText[i].strip() for i in sortedIdxs]
    currentDF = pd.DataFrame({cols[0]: cols[1:]})

    # concatenate *original* data frame with the *current* data
    # frame (we do this to handle columns that may have a varying
    # number of rows)
    df = pd.concat([df, currentDF], axis=1)

# replace NaN values with an empty string and then show a nicely
# formatted version of our multi-column OCR'd text
df.fillna("", inplace=True)
print(tabulate(df, headers="keys", tablefmt="psql"))