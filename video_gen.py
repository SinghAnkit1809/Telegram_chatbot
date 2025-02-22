import os
import math
import time
import re
import json
import random
import requests
import asyncio
import edge_tts
import sys
from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
import numpy as np
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont

class VideoGenerator:
    def __init__(self):
        self.model = "openai"
        self.width = 1080
        self.height = 1920
        self.target_duration = 60
        self.max_segment_duration = 5
        self.num_segments = math.ceil(self.target_duration / self.max_segment_duration)
        self.speech_rate = "-10%"

    def download_image(self, prompt, seed):
        """Download image from pollinations.ai with proper error handling"""
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://pollinations.ai/p/{encoded_prompt}?width={self.width}&height={self.height}&seed={seed}&model=flux&nologo=true"
        
        for attempt in range(3):
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to download image after 3 attempts: {str(e)}")
                time.sleep(1)

    async def generate_video(self, topic, progress_callback=None):
        """Main video generation workflow"""
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            try:
                # Story generation
                if progress_callback:
                    await progress_callback("📝 Crafting your story...")
                story, segments = self._generate_story(topic)
                
                # Image generation
                if progress_callback:
                    await progress_callback("🎨 Painting your vision...")
                image_prompts = self._create_image_prompts(story, segments)
                images = [np.array(self.download_image(p, random.randint(0, 999999))) 
                         for p in image_prompts]
                
                # Audio generation
                if progress_callback:
                    await progress_callback("🔊 Recording narration...")
                audio_path = await self._generate_audio(segments, temp_path)
                
                # Get video bytes before cleanup
                video_bytes = self.compile_video(images, segments, audio_path, temp_path)
                return video_bytes
                
            except Exception as e:
                raise RuntimeError(f"Generation failed: {str(e)}")

    def _generate_story(self, topic):
        """Generate story segments with proper pacing"""
        prompt = f"""Create a {self.target_duration}-second story about {topic} with:
        - Simple English (A2/B1 level)
        - Natural pauses every {self.max_segment_duration} seconds
        - Concrete examples
        - Clear cause-effect flow
        - {self.num_segments} segments"""
        
        response = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": "Expert storyteller"},
                    {"role": "user", "content": prompt}
                ],
                "model": self.model,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return self._split_text(response.text.strip())

    def _split_text(self, text):
        """Split text into natural speaking segments"""
        segments = []
        remaining = text
        target_length = len(text) // self.num_segments
        
        while remaining and len(segments) < self.num_segments:
            for punctuation in ['. ', '! ', '? ', '; ', ', ']:
                split_at = remaining.rfind(punctuation, 0, target_length + 50)
                if split_at != -1:
                    segments.append(remaining[:split_at+1].strip())
                    remaining = remaining[split_at+1:].strip()
                    break
            else:
                segments.append(remaining[:target_length].strip())
                remaining = remaining[target_length:].strip()
        
        if remaining:
            segments[-1] += " " + remaining
        return text, segments[:self.num_segments]

    def _create_image_prompts(self, story, segments):
        """Generate consistent visual prompts for each segment"""
        prompt = f"""Create image prompts for these story segments:
        {json.dumps(segments)}
        Rules:
        - Maintain character consistency
        - Cartoonish digital art style
        - Max 120 characters
        - Use descriptive terms, no pronouns"""
        
        response = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": "Visual continuity expert"},
                    {"role": "user", "content": prompt}
                ],
                "model": self.model,
                "jsonMode": True
            },
            timeout=60
        )
        
        try:
            return [p[:120] for p in json.loads(response.text)['prompts']]
        except:
            return [f"{seg[:100]} cartoon digital art" for seg in segments]

    async def _generate_audio(self, segments, temp_path):
        """Generate and combine audio segments"""
        audio_files = []
        for i, text in enumerate(segments):
            path = temp_path / f"audio_{i}.mp3"
            communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
            await communicate.save(str(path))
            audio_files.append(path)
        
        combined = concatenate_audioclips([AudioFileClip(str(f)) for f in audio_files])
        final_path = temp_path / "final_audio.mp3"
        combined.write_audiofile(str(final_path))
        return final_path

    def compile_video(self, images, segments, audio_path, temp_path):
        """Create final video file and return its bytes"""
        try:
            audio = AudioFileClip(str(audio_path))
            clips = []
            
            segment_duration = audio.duration / len(segments)
            for img, text in zip(images, segments):
                img_with_caption = self._add_captions(img, text)
                clip = ImageClip(img_with_caption).set_duration(segment_duration)
                clips.append(clip)
            
            video = concatenate_videoclips(clips).set_audio(audio)
            video_path = temp_path / "final_video.mp4"
            
            # Write to in-memory bytes buffer
            video_bytes = BytesIO()
            video.write_videofile(
                str(video_path),
                fps=24,
                threads=4,
                preset='ultrafast',
                ffmpeg_params=['-movflags', '+faststart'],
                logger=None  # Disable progress logs
            )
            
            # Read generated file back into memory
            with open(video_path, 'rb') as f:
                video_bytes.write(f.read())
            
            video_bytes.seek(0)
            return video_bytes
            
        except Exception as e:
            raise RuntimeError(f"Video compilation failed: {str(e)}")
        finally:
            # Ensure MoviePy objects are closed
            if 'video' in locals():
                video.close()
            if 'audio' in locals():
                audio.close()

    def _add_captions(self, image_array, text):
        """Add dynamic captions to images"""
        img = Image.fromarray(image_array)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial_Bold.ttf", 60)
        except:
            font = ImageFont.load_default(60)
        
        # Smart text wrapping
        words = text.split()
        lines = []
        current_line = []
        max_width = self.width - 200
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            w, _ = draw.textsize(test_line, font=font)
            if w > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        
        # Center text vertically
        total_height = sum(font.getsize(line)[1] for line in lines) + 10*(len(lines)-1)
        y = (self.height - total_height) // 2
        
        for line in lines:
            w, h = draw.textsize(line, font=font)
            x = (self.width - w) // 2
            draw.text((x, y), line, font=font, fill="#FFFF00", stroke_width=3, stroke_fill="#000000")
            y += h + 10
        
        return np.array(img) 