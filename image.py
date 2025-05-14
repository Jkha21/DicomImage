# Prerequisites:
# 1. Install Python (if not already installed) from python.org
# 2. Open your command prompt or PowerShell and install the required libraries:
#    pip install pydicom Pillow numpy

import pydicom
import numpy as np
from PIL import Image
import os

def extract_frames_from_dicom(dicom_filepath, base_output_folder="extracted_images"):
    """
    Extracts individual frames from a multi-frame or single-frame DICOM file
    if it contains pixel data, saving them to a subfolder named after the source file.
    Handles files without pixel data gracefully.

    Args:
        dicom_filepath (str): The full path to the input DICOM file (.dcm).
        base_output_folder (str): The main folder where extracted images will be saved
                                  in subfolders. Defaults to "extracted_images" in
                                  the directory where the script is run.
    """
    # Print the file being processed for clarity, using just the base filename
    print(f"Processing file: {os.path.basename(dicom_filepath)}")

    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_filepath)

        # --- Check if the file contains pixel data using its standard tag (7FE0, 0010) ---
        # This check determines if the file is an image file vs. a metadata/reference file.
        # If this tag is missing, the file does not contain the actual image pixels.
        if (0x7fe0, 0x0010) not in ds:
            print(f"Info: '{os.path.basename(dicom_filepath)}' is a DICOM file but does not contain standard pixel data (tag 7FE0,0010 missing). Skipping.")
            # Check if it's likely a file referencing images (like a DICOMDIR or Key Object Selection)
            # using the correct hexadecimal notation for the Referenced Image Sequence tag (0008, 1140)
            # This tag means the file *points to* images elsewhere, it doesn't *contain* them.
            if (0x0008, 0x1140) in ds:
                 print("      It appears to contain a Referenced Image Sequence, pointing to other image files.")
            return # Exit the function if no pixel data is found

        # If pixel data IS present, get it as a NumPy array
        pixel_array = ds.pixel_array

        # --- Determine the specific output folder for this DICOM file ---
        base_filename_without_ext = os.path.splitext(os.path.basename(dicom_filepath))[0]
        specific_output_folder_name = f"{base_filename_without_ext}_extracted" # Create a folder name based on the source file

        # Ensure the specific output folder exists within the base output folder
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) # Get script's directory
        except NameError:
             # Handle cases where __file__ might not be defined (e.g., in some interactive environments)
             script_dir = os.getcwd() # Use current working directory as a fallback

        # Construct the full path to the specific output folder
        specific_output_path = os.path.join(script_dir, base_output_folder, specific_output_folder_name)

        # Create the folder if it doesn't exist
        if not os.path.exists(specific_output_path):
            os.makedirs(specific_output_path)

        # Process based on the dimensions of the pixel array
        if pixel_array.ndim == 3:
            # 3D array: likely (number_of_frames, height, width) for grayscale multi-frame
            num_frames = pixel_array.shape[0]
            print(f"Found {num_frames} frames in this multi-frame DICOM file.")

            for i in range(num_frames):
                frame = pixel_array[i]

                # Scale pixel data to 8-bit for common image formats (adjust if needed)
                # This simple scaling works for many grayscale images.
                # More sophisticated windowing/leveling might be needed for medical display.
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.float64)
                    # Avoid division by zero if max and min are the same (flat images)
                    min_val = np.min(frame)
                    max_val = np.max(frame)
                    if max_val != min_val:
                         frame = 255 * (frame - min_val) / (max_val - min_val)
                    else:
                         frame = np.zeros_like(frame, dtype=np.float64) # Handle flat images
                    frame = frame.astype(np.uint8)

                # Create a PIL Image object
                # Check if grayscale (2D array) or color (3D array with last dim = 3 or 4)
                if frame.ndim == 2:
                     image = Image.fromarray(frame, 'L') # 'L' for grayscale
                elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
                     # Assume RGB or RGBA
                     image = Image.fromarray(frame)
                else:
                     print(f"Warning: Unexpected frame dimensions {frame.shape} for frame {i}. Skipping frame.")
                     continue


                # Define output filename within the specific subfolder
                output_filename = os.path.join(specific_output_path, f"frame_{i:04d}.png") # Saves as PNG

                # Save the image
                image.save(output_filename)
                # print(f"Saved {os.path.basename(output_filename)}") # Optional: uncomment to see each frame saved


            print(f"Extraction complete for '{os.path.basename(dicom_filepath)}'. Images saved to '{os.path.join(base_output_folder, specific_output_folder_name)}'")

        elif pixel_array.ndim == 4:
             # 4D array: likely (number_of_frames, height, width, channels) for color or multi-channel multi-frame
             num_frames = pixel_array.shape[0]
             print(f"Found {num_frames} frames in this multi-frame DICOM file (potentially color/multi-channel).")

             for i in range(num_frames):
                 frame = pixel_array[i]

                 # Scale if not already 8-bit (handle multi-channel scaling if necessary)
                 if frame.dtype != np.uint8:
                     # Simple scaling for multi-channel might need adjustment
                     frame = frame.astype(np.float64)
                     min_val = np.min(frame)
                     max_val = np.max(frame)
                     if max_val != min_val:
                         frame = 255 * (frame - min_val) / (max_val - min_val)
                     else:
                          frame = np.zeros_like(frame, dtype=np.float64) # Handle flat images
                     frame = frame.astype(np.uint8)

                 # Create a PIL Image object (PIL handles the channels)
                 image = Image.fromarray(frame)

                 # Define output filename within the specific subfolder
                 output_filename = os.path.join(specific_output_path, f"frame_{i:04d}.png") # Saves as PNG

                 # Save the image
                 image.save(output_filename)
                 # print(f"Saved {os.path.basename(output_filename)}") # Optional: uncomment to see each frame saved


             print(f"Extraction complete for '{os.path.basename(dicom_filepath)}'. Images saved to '{os.path.join(base_output_folder, specific_output_folder_name)}'")


        elif pixel_array.ndim == 2:
             # 2D array: likely a single-frame grayscale image
             print(f"Found a single-frame DICOM image.")
             frame = pixel_array

             # Scale pixel data to 8-bit if needed
             if frame.dtype != np.uint8:
                 frame = frame.astype(np.float64)
                 min_val = np.min(frame)
                 max_val = np.max(frame)
                 if max_val != min_val:
                    frame = 255 * (frame - min_val) / (max_val - min_val)
                 else:
                    frame = np.zeros_like(frame, dtype=np.float64) # Handle flat images
                 frame = frame.astype(np.uint8)

             # Create a PIL Image object ('L' for grayscale)
             image = Image.fromarray(frame, 'L')

             # Define output filename within the specific subfolder
             output_filename = os.path.join(specific_output_path, f"single_frame.png") # Saves as PNG
             image.save(output_filename)
             print(f"Saved single frame to '{os.path.join(base_output_folder, specific_output_folder_name, 'single_frame.png')}'")

        else:
            print(f"Warning: Unexpected pixel array dimensions ({pixel_array.ndim}) in '{os.path.basename(dicom_filepath)}'. Cannot process as image data.")


    except pydicom.errors.InvalidDicomError:
        # Handles errors specific to invalid DICOM format
        print(f"Error: '{os.path.basename(dicom_filepath)}' does not appear to be a valid DICOM file. Skipping.")
    except FileNotFoundError:
        # Handles case where the input file path is incorrect
        print(f"Error: Input file not found at '{dicom_filepath}'.")
    except Exception as e:
        # Catches any other unexpected errors during processing
        print(f"An unexpected error occurred while processing '{os.path.basename(dicom_filepath)}': {e}")


# --- How to use the script ---
if __name__ == "__main__":
    # --- Option 1: Process a single known DICOM image file ---
    # Uncomment the following two lines and modify the path to your specific DICOM image file
    # dicom_file_path = "path/to/your/actual_image_file.dcm" # <<< CHANGE THIS
    # extract_frames_from_dicom(dicom_file_path)

    # --- Option 2: Process all .dcm files in a folder ---
    # Uncomment the following line and modify the path to the folder containing your DICOM files
    # This is useful if you have multiple .dcm files and aren't sure which are the image files.
    # The script will process all .dcm files found, skipping those without pixel data
    # and extracting images from those that do.
    dicom_folder_path = "path of file" # <<< CHANGE THIS

    if not os.path.isdir(dicom_folder_path):
        print(f"Error: The specified folder does not exist: {dicom_folder_path}")
    else:
        print(f"Scanning folder: {dicom_folder_path}")
        # List files and sort them to process in a consistent order (optional but good practice)
        # Use try-except here in case os.listdir fails
        try:
            filenames = sorted(os.listdir(dicom_folder_path))
        except Exception as e:
            print(f"Error listing files in folder {dicom_folder_path}: {e}")
            filenames = [] # Set to empty list to prevent further errors

        for filename in filenames:
            # Process files ending with .dcm (case-insensitive)
            if filename.lower().endswith(".dcm"):
                file_path = os.path.join(dicom_folder_path, filename)
                # Call the extraction function for each .dcm file
                extract_frames_from_dicom(file_path)
            # Optional: print message for non-dcm files
            # else:
            #    print(f"Skipping non-DICOM file: {filename}")

        print("\nFinished processing all .dcm files in the specified folder.")