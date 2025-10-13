from PIL import Image
import numpy as np
import io
import base64
import pandas as pd 

def base64encoding(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def csv_encoding(df: pd.DataFrame):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False) # write into the csv buffer instead of actual disk
    return base64.b64encode(csv_buffer.getvalue().encode()).decode() # get the base64 encoded csv