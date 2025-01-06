
import os
import shutil
from PIL import Image


def resize_images_in_folder(folder_path, max_dimension=640):
    """
    Resize all images in the specified folder such that the larger dimension is at most max_dimension.
    Maintains aspect ratio and image quality. Non-resized images are copied to the output folder.

    Args:
        folder_path (str): Path to the folder containing the images.
        max_dimension (int): Maximum size of the larger dimension (default is 640).
    """
    # Create an output folder in the same directory as the input folder
    output_folder = os.path.join(os.path.dirname(folder_path), "resized")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only image files
        if not os.path.isfile(file_path) or not filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')):
            continue

        with Image.open(file_path) as img:
            width, height = img.size

            # Check if resizing is needed
            if max(width, height) > max_dimension:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int((max_dimension / width) * height)
                else:
                    new_height = max_dimension
                    new_width = int((max_dimension / height) * width)

                # Resize the image
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                output_path = os.path.join(output_folder, filename)

                # Save resized image with quality preserved
                img_resized.save(output_path, quality=95)
                print(f"Resized and saved: {output_path}")
            else:
                # Copy the original image to the output folder
                output_path = os.path.join(output_folder, filename)
                shutil.copy(file_path, output_path)
                print(f"Copied without resizing: {output_path}")


def main():
    # Set the path to your images folder here
    folder_path = "/home/guilh/data_tese/Machine_Learning/R4f_final_seg/images"  # Change this to your folder path
    max_dimension = 640  # Maximum dimension for resizing

    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    resize_images_in_folder(folder_path, max_dimension)


if __name__ == "__main__":
    main()
