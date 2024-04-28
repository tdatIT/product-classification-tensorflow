import requests
import uuid
import os


def download_image(pic_url):
    image_name = str(uuid.uuid4()) + '.jpg'
    basepath = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.join('uploads', image_name)
    with open(image_path, 'wb') as handle:
        response = requests.get(pic_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

    return os.path.join(basepath, image_path)
