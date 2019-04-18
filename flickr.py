import os

import time
import traceback

import flickrapi
from urllib.request import urlretrieve

import sys
from retry import retry

# please change below 2
flickr_api_key = "402....."
secret_key = "ba8..."
image_dir='./images/'
num_photos = 100
keywords = ['sakura', 'sunflower', 'rose']


@retry()
def get_photos(url, filepath):
    urlretrieve(url, filepath)
    time.sleep(1)


def download(keyword, photos):
    try:
        if not os.path.exists(image_dir + keyword):
            os.makedirs(image_dir + keyword)

        i = 0
        for photo in photos['photo']:
            url_q = photo['url_q']
            filepath = image_dir + keyword + '/' + photo['id'] + '.jpg'
            print('downloading {}: {}->{}'.format(i, url_q, filepath))
            get_photos(url_q, filepath)
            i+=1

    except Exception as e:
        traceback.print_exc()

def main():

    flicker = flickrapi.FlickrAPI(flickr_api_key, secret_key, format='parsed-json')
    for keyword in keywords:
        print('*** {} ***'.format(keyword))
        response = flicker.photos.search(
            text=keyword,
            per_page=num_photos,
            media='photos',
            sort='relevance',
            safe_search=1,
            extras='url_q,license'
        )
        photos = response['photos']
        download(keyword, photos)

if __name__ == '__main__':
    main()
