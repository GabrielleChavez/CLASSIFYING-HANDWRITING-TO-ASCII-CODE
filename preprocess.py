import os
from PIL import Image
import xml.etree.ElementTree as ET
import html

def stitch_images(path: str, output_path: str, padding_width: int) -> None:

    """
    Stitch all PNG images in the given folder together into a single image side by side.
    The images with smaller heights are padded to the maximum height with whitespace on top and bottom and stitched side by side with padding in between.
    The final stitched image is saved to the output path.

    :param path: The path to the folder containing the PNG images.
    :param output_path: The path to save the stitched image.
    :param padding_width: The width of the padding between images.

    :raises ValueError: If there are no PNG images in the given folder.
    
    """

    # Gather all PNG images
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.png')]

    if not image_paths:
        raise ValueError("No .png images in the given folder.")

    # Open images and sort them based on the suffix after the last '-'
    images_dict = {os.path.basename(p)[:-4].split('-')[-1]: Image.open(p) for p in image_paths}
    names = list(images_dict.keys())
    names.sort()
    images = [images_dict[name] for name in names]

    # Find the maximum height among all images
    _, heights = zip(*(img.size for img in images))
    max_height = max(heights)

    # Create new images that are padded to the max height, centering the original image
    padded_images = []
    for img in images:
        w, h = img.size
        # Create a new white background image with max height
        bg = Image.new('RGB', (w, max_height), color=(255, 255, 255))
        # Calculate vertical offset to center the image
        top_offset = (max_height - h) // 2
        # Paste the original image onto the background
        bg.paste(img, (0, top_offset))
        padded_images.append(bg)

    # Compute the total width with padding
    total_width = sum(img.size[0] for img in padded_images) + padding_width * (len(padded_images) - 1)

    # Create the final stitched image
    stitched_image = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # Paste all images side by side with padding
    x_offset = 0
    for i, img in enumerate(padded_images):
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
        if i < len(padded_images) - 1:
            x_offset += padding_width

    # Save the stitched image
    stitched_image.save(output_path)


def stitch_all_IAM_images(lines_folder: str, output_folder: str) -> None:
    """
    Stitch all the images in the IAM lines folder into a single image and save it in the output folder.
    Creates a new folder in the same format as the IAM lines folder but with the single stitched image for each one.

    :param lines_folder: The path to the IAM lines folder.
    :param output_folder: The path to save the stitched images
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for outer_folder in os.listdir(lines_folder):
        # Iterating over the outmost folders in the IAM lines
        outer_folder_path = os.path.join(lines_folder, outer_folder)
        if not os.path.exists(os.path.join(output_folder, outer_folder)):
            os.makedirs(os.path.join(output_folder, outer_folder))
        for inner_folder in os.listdir(outer_folder_path):
            # Iterating over the inner folders in the IAM lines
            inner_folder_path = os.path.join(outer_folder_path, inner_folder)
            output_path = os.path.join(output_folder, outer_folder, inner_folder)
            if not os.path.exists(os.path.join(output_folder, outer_folder, inner_folder)):
                os.makedirs(output_path)
            stitch_images(inner_folder_path, output_path + "/" + inner_folder + '.png', 50)

def map_transcriptions(lines_folder: str, xml_folder: str) -> None:

    """
    Map the transcriptions from the XML files to the corresponding image folders in the IAM lines folder.
    Create a new text file in each image folder with the transcription.
    
    :param lines_folder: The path to the IAM lines folder.
    :param xml_folder: The path to the folder containing the XML files.

    """

    for outer_file in os.listdir(lines_folder):
        for inner_file in os.listdir(os.path.join(lines_folder, outer_file)):
            xml_file_path = os.path.join(xml_folder, inner_file + '.xml')
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            machine_printed_part = root.find("machine-printed-part")

            file_transcription = ''
            if machine_printed_part is not None:
                lines = machine_printed_part.findall("machine-print-line")
                for line in lines:
                    text = line.get("text", "")
                    file_transcription += text + " "
            else:
                raise ValueError("No transcription found in the XML file.")
            
            file_transcription = html.unescape(file_transcription)
            
            with open(os.path.join(lines_folder, outer_file, inner_file, f"{inner_file}-transcription.txt"), "w") as f:
                f.write(file_transcription)

def get_vocabulary(lines_folder: str) -> set:

    """
    Gets the individual characters in the transcriptions of the processed IAM lines folder.

    :param lines_folder: The path to the processed IAM lines folder. 

    :return: A set of characters
    """

    vocabulary = set()
    for outer_folder in os.listdir(lines_folder):
        outer_folder_path = os.path.join(lines_folder, outer_folder)
        for inner_folder in os.listdir(outer_folder_path):
            inner_folder_path = os.path.join(outer_folder_path, inner_folder)
            transcription_file = os.path.join(inner_folder_path, f"{inner_folder}-transcription.txt")
            with open(transcription_file, "r") as f:
                transcription = f.read()
                chars = set(transcription)
                vocabulary.update(chars)

    return vocabulary

def get_max_height_width(lines_folder: str) -> tuple:
        max_height = 0
        max_width = 0
        for outer_folder in os.listdir(lines_folder):
            outer_folder_path = os.path.join(lines_folder, outer_folder)
            for inner_folder in os.listdir(outer_folder_path):
                inner_folder_path = os.path.join(outer_folder_path, inner_folder)
                image_path = os.path.join(inner_folder_path, f"{inner_folder}.png")
                image = Image.open(image_path)  
                image_size = image.size
                height = image_size[1]
                width = image_size[0]

                if height > max_height:
                    max_height = height
                if width > max_width:
                    max_width = width
        
        return max_width, max_height

def pad_to_max(img_path: str, width: int, height: int):
    img = Image.open(img_path)
    new_img = Image.new('RGB', (width, height), color=(255, 255, 255))
    new_img.paste(img, (0, 0))
    return new_img

def pad_all_to_max(lines_path: str, width: int, height: int) -> None:
    for outer_folder in os.listdir(lines_path):
        outer_folder_path = os.path.join(lines_path, outer_folder)
        for inner_folder in os.listdir(outer_folder_path):
            inner_folder_path = os.path.join(outer_folder_path, inner_folder)
            image_path = os.path.join(inner_folder_path, f"{inner_folder}.png")
            new_img = pad_to_max(image_path, width, height)
            new_img.save(image_path)

if __name__ == '__main__':

    width, height = get_max_height_width("lines_processed")
    pad_all_to_max("lines_processed_padded", width, height)
    
    # How I used these to get the processed IAM lines folder:
    # stitch_all_IAM_images("lines", "lines_processed")    
    # map_transcriptions("lines_processed", "xml_files")
    # vocabulary = get_vocabulary("lines_processed")
    # with open("vocabulary.txt", "w") as f:
    #     for char in vocabulary:
    #         f.write(char + "\n")

    pass




