#telegram bot image_gen.py
import requests
from urllib.parse import quote

def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    model: str = "Flux",
    seed: int = None,
    enhance: bool = True,
    nologo: bool = True,
    nofeed: bool = True,
    safe: bool = True,
    negative_prompt: str = None
) -> str:
    """Generate image URL using pollinations.ai API"""
    base_url = "https://image.pollinations.ai/prompt/"
    params = {
        'width': width,
        'height': height,
        'model': model,
        'Enhance': str(enhance).lower(),
        'nologo': str(nologo).lower(),
        'nofeed': str(nofeed).lower(),
        'safe': str(safe).lower(),
        'seed': seed
    }
    
    if negative_prompt:
        params['negative_prompt'] = negative_prompt
    
    # Build query string
    query = '&'.join([f'{k}={v}' for k, v in params.items() if v is not None])
    encoded_prompt = quote(prompt)
    
    return f"{base_url}{encoded_prompt}?{query}" 