import base64
import os
import traceback
from typing import Optional
from datetime import datetime
import requests
import uuid
from io import BytesIO

from PIL import Image


def base64_to_image(base64_string: str,
                    output_path: Optional[str] = None,
                    filename: Optional[str] = None) -> str:
    try:
        # Process base64 string, remove possible prefix (e.g. "data:image/png;base64,")
        if ',' in base64_string:
            header, base64_data = base64_string.split(',', 1)
        # Extract file extension from header
            if 'image/' in header:
                image_format = header.split('image/')[1].split(';')[0]
                if image_format == 'jpeg':
                    image_format = 'jpg'
            else:
                image_format = 'png'  # Default format
        else:
            base64_data = base64_string
            image_format = 'png'  # Default format

        # Decode base64 string
        image_data = base64.b64decode(base64_data)

        # Set output path
        if output_path is None:
            output_path = 'images'

        # Create output directory (if not exists)
        os.makedirs(output_path, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"image_{timestamp}_{unique_id}.{image_format}"
        elif not filename.endswith(f'.{image_format}'):
            filename = f"{filename}.{image_format}"

        # Complete file path
        full_path = os.path.join(output_path, filename)

        # Save image file
        with open(full_path, 'wb') as f:
            f.write(image_data)

        print(f"Image successfully saved to: {full_path}")
        return full_path

    except Exception as e:
        raise IOError(f"Error occurred while saving image: {str(e)}")


def image_to_base64(image_path: str, max_size: int = 1024, quality: int = 95, format: str = 'JPEG') -> str:
    try:
        # Open image
        with Image.open(image_path) as img:
        # Compress image
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Handle transparent background
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

        # Save to memory buffer
            buffer = BytesIO()
            img.save(buffer, format=format, quality=quality, optimize=True)
            compressed_data = buffer.getvalue()

        # Encode to base64
            base64_encoded = base64.b64encode(compressed_data).decode('utf-8')

        # Print compression information
            original_size = os.path.getsize(image_path)
            compressed_size = len(compressed_data)

            return f"data:image;base64,{base64_encoded}"

    except Exception as e:
        print(f"Image compression failed {image_path}: {str(e)}, using original image")
        # If compression fails, use original image
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            return base64.b64encode(binary_data).decode('utf-8')


def png_to_jpg_compressed(input_path: str, output_path: str = None, quality: int = 95):
    """
    Losslessly compress PNG image to JPG format

    Args:
        input_path (str): Input PNG image path
        output_path (str, optional): Output JPG image path, default is original path with '_compressed' suffix
        quality (int): JPG image quality, range 1-95, default 95

    Returns:
        str: Path of compressed JPG image

    Raises:
        FileNotFoundError: When input file does not exist
        IOError: When file processing fails
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Check if file is PNG format
    if not input_path.lower().endswith('.png'):
        return Image.open(input_path)

    # If no output path is specified, use default path
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.jpg"

    # Open PNG image
    with Image.open(input_path) as img:
        # Handle transparent background, convert to RGB mode
        print("image mode: ", img.mode)
        if img.mode in ('RGBA', 'LA'):
        # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
        # If there is transparency, paste onto white background
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
        elif img.mode == 'P':
        # If in palette mode, first convert to RGBA then process
            img = img.convert('RGBA')
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
        # Ensure image mode is RGB
            img = img.convert('RGB')

        # Save as JPG format
        # img.save(output_path, 'JPEG', quality=quality, optimize=True)

    return img


def pil_image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL.Image object to base64 string

    Args:
        image (Image.Image): PIL.Image object
        format (str): Image format, default is 'PNG'

    Returns:
        str: Base64 encoded image string

    Raises:
        ValueError: When image is not a PIL.Image object
    """
    # Check if image is PIL.Image object
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be PIL.Image object")

    # Save image to bytes object in memory
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()

    # Encode bytes object to base64 string
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    return f"data:image;base64,{base64_string}"


def image_url_to_base64(image_url: str) -> Optional[dict]:
    """
    Convert image URL to base64 string and get image dimensions

    Args:
        image_url: URL address of the image

    Returns:
        Dictionary containing base64 string and dimension information, returns None if failed
        Format: {
            'base64': 'data:image/jpeg;base64,...',
            'width': 800,
            'height': 600
        }
    """
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        print("request image response: ", response.content)
        # Get image content
        image_content = response.content

        # Get image format
        content_type = response.headers.get('content-type', '')
        if 'image/' not in content_type:
        # Try to determine from URL suffix
            if image_url.lower().endswith('.png'):
                content_type = 'image/png'
            elif image_url.lower().endswith('.jpg') or image_url.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif image_url.lower().endswith('.gif'):
                content_type = 'image/gif'
            elif image_url.lower().endswith('.webp'):
                content_type = 'image/webp'
            else:
                content_type = 'image/jpeg'  # Default

        # Use PIL to get image dimensions
        from io import BytesIO
        img = Image.open(BytesIO(image_content))
        width, height = img.size

        # Convert to base64
        base64_content = base64.b64encode(image_content).decode('utf-8')

        # Return dictionary containing base64 and dimension information
        return {
            'base64': f"data:{content_type};base64,{base64_content}",
            'width': width,
            'height': height
        }

    except Exception as e:
        print(f"Failed to convert image: {e}")
        traceback.print_exc()
        return None


def download_image_from_url(image_url: str, output_path: str, filename: Optional[str] = None) -> str:
    """
    Download image from URL and save locally

    Args:
        image_url: URL address of the image
        output_path: Local directory path to save the image
        filename: Optional, custom filename (without extension), auto-generated if None

    Returns:
        Complete file path after saving

    Raises:
        IOError: When downloading or saving image fails
    """
    try:
        # Send request to get image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Get image content
        image_content = response.content

        # Get image format
        content_type = response.headers.get('content-type', '')
        if 'image/' not in content_type:
        # Try to determine from URL suffix
            if image_url.lower().endswith('.png'):
                image_format = 'png'
            elif image_url.lower().endswith('.jpg') or image_url.lower().endswith('.jpeg'):
                image_format = 'jpg'
            elif image_url.lower().endswith('.gif'):
                image_format = 'gif'
            elif image_url.lower().endswith('.webp'):
                image_format = 'webp'
            elif image_url.lower().endswith('.bmp'):
                image_format = 'bmp'
            elif image_url.lower().endswith('.svg'):
                image_format = 'svg'
            else:
        # Try to determine format from image content
                try:
                    from PIL import Image
                    from io import BytesIO
                    img = Image.open(BytesIO(image_content))
                    image_format = img.format.lower()
                    if image_format == 'jpeg':
                        image_format = 'jpg'
                except:
                    image_format = 'jpg'  # Default format
        else:
        # Extract format from content-type
            image_format = content_type.split('image/')[1].split(';')[0]
            if image_format == 'jpeg':
                image_format = 'jpg'

        # Create output directory (if not exists)
        os.makedirs(output_path, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"image_{timestamp}_{unique_id}.{image_format}"
        else:
            filename = f"{filename}.{image_format}"

        # Complete file path
        full_path = os.path.join(output_path, filename)

        # Save image file
        with open(full_path, 'wb') as f:
            f.write(image_content)

        print(f"Image successfully downloaded and saved to: {full_path}")
        return full_path

    except requests.exceptions.RequestException as e:
        raise IOError(f"Failed to download image: {str(e)}")
    except Exception as e:
        raise IOError(f"Error occurred while saving image: {str(e)}")


if __name__ == '__main__':
    image_path = "test_image/image1.jpg"


    img_str = image_to_base64(image_path)
    with open("test_image/test_image_string1.txt", "w") as outf:
        outf.write(img_str)