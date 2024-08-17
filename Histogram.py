import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd

# Read the data from the CSV file
df = pd.read_csv(r'C:\Users\Rodrigo\OneDrive\Escritorio\UP\IntelligentsAgents\archive\HAM10000_metadata.csv')

# Create a list to store the histograms
histograms = []
i = 0
# Iterate over each row in the CSV file
for index, row in df.iterrows():
    # Get the image file name from the CSV row
    image_file_name = row['image_id'] 
    i += 1
    print(i)

    # Load the image
    img = io.imread(r"C:\Users\Rodrigo\OneDrive\Escritorio\UP\IntelligentsAgents\archive\images\{}.jpg".format(image_file_name))
    
    # Reshape the image into a 3D array of RGB values
    img_reshape = img.reshape(-1, 3) # This reshapes the image array into a 2D array where each row represents a pixel.
    
    # Bin the RGB values into 8 bins each (24 bins total)
    bins = np.linspace(0, 256, 9)  # 8 bins from 0 to 256
    binned_rgb = np.digitize(img_reshape, bins) - 1  # subtract 1 to get 0-based indices
    
    # Calculate the histogram for each color channel
    hist_r = np.bincount(binned_rgb[:, 0], minlength=8) # These lines count the number of pixels in each bin for each color channel.
    hist_g = np.bincount(binned_rgb[:, 1], minlength=8) # bincount counts occurrences of each integer value in the input array.
    hist_b = np.bincount(binned_rgb[:, 2], minlength=8) # minlength=8 ensures we get counts for all 8 bins, even if some are empty.
    
    """if i < 5:
        histogram = np.concatenate((hist_r, hist_g, hist_b))
        print(histogram)
        plt.bar(range(8), hist_r, color='red', alpha=1, label='Red')
        plt.bar(range(8), hist_g, color='green', alpha=1, label='Green')
        plt.bar(range(8), hist_b, color='blue', alpha=1, label='Blue')
        plt.xlabel('Bin Index')
        plt.ylabel('Frequency')
        plt.title('RGB Histograms')
        plt.legend()
        plt.show()"""
    # Concatenate the histograms
    histogram = np.concatenate((hist_r, hist_g, hist_b))

    # Append the histogram to the list
    histograms.append(histogram)

# Convert the list of histograms to a NumPy array
histograms_array = np.array(histograms)

# Save the histograms in a do
np.save('histogram_data.npy', histograms_array)

print("--------Array----------")
print(histograms_array)
print(len(histograms_array))
print("--------")
print(histograms)
print(len(histograms))
# You can now use the histograms_array for further analysis or visualization