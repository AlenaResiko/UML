"""
NOTE
Make use of the lyricsgenius package to download lyrics from the Genius API.
TODO YOUR_API_KEY should be stored in a .env file in the root directory.

Create api https://genius.com/api-clients

pip install lyricsgenius https://pypi.org/project/lyricsgenius/
"""

import os

MUSIC_DIR = os.path.join(os.environ["DATA_DIR"], "music/")
