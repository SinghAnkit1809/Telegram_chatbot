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
import numpy as np
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
from tempfile import TemporaryDirectory

class VideoGenerator:
    def __init__(self):
        self.model = "openai"
        # self.width = 1080
        # self.height = 1920
        # self.target_duration = 100
        # self.max_segment_duration = 5
        self.width = 720
        self.height = 1280
        self.target_duration = 45
        self.max_segment_duration = 4
        self.num_segments = math.ceil(self.target_duration / self.max_segment_duration)
        self.speech_rate = "-10%"

    def download_image(self, prompt, seed, max_retries=3):
        """Download image from pollinations.ai with proper error handling"""
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://pollinations.ai/p/{encoded_prompt}?width={self.width}&height={self.height}&seed={seed}&model=flux&nologo=true"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download image after {max_retries} attempts: {str(e)}")
                time.sleep(2)
                continue

    async def generate_video(self, topic, progress_callback=None):
        """Main video generation workflow"""
        # Clear any existing memory
        if hasattr(self, 'last_video'):
            del self.last_video
        import gc; gc.collect()
        
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            try:
                # Story generation
                if progress_callback:
                    await progress_callback("📝 Crafting your story...")
                story, segments = self._generate_story(topic)
                
                # Image generation
                if progress_callback:
                    await progress_callback("🎨 Creating visuals...")
                image_prompts = self._create_image_prompts(story, segments)
                images = []
                
                for i, prompt in enumerate(image_prompts):
                    for attempt in range(3):
                        try:
                            seed = random.randint(1, 999999)
                            image = self.download_image(prompt, seed)
                            images.append(np.array(image))
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise RuntimeError(f"Failed to generate image {i+1}: {str(e)}")
                            await asyncio.sleep(1)
                
                # Audio generation
                if progress_callback:
                    await progress_callback("🔊 Recording narration...")
                audio_path = await self._generate_audio(segments, temp_path)
                
                # Video compilation
                if progress_callback:
                    await progress_callback("🎥 Compiling final video...")
                video_bytes = await self._compile_video(images, segments, audio_path, temp_path)
                
                return video_bytes
                
            except Exception as e:
                raise RuntimeError(f"Video generation failed: {str(e)}")

    def _generate_story(self, topic):
        """Generate story segments with proper pacing"""
        prompt = f"""Write a simple, engaging story about {topic} that can be narrated in exactly {self.target_duration} seconds. Follow these rules:
        1. Use simple English (A2/B1 level) with short sentences
        2. Include natural speaking pauses between ideas
        3. Use concrete words over abstract concepts
        4. Limit complex vocabulary (e.g. use 'make' instead of 'fabricate')
        5. Structure with clear cause-effect relationships
        6. Use everyday examples readers can relate to
        
        Example good sentence: "When the rain didn't stop, Mia knew she had to move her garden to higher ground."
        Example bad sentence: "The precipitation persisting, Mia was compelled to relocate her horticultural project elevationally."
        
        Make exactly {self.num_segments} natural segments with oral storytelling flow. Return ONLY the story."""
        
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
        """Split text into well-paced segments"""
        segments = []
        remaining = text
        target_length = len(text) // self.num_segments
        
        while remaining and len(segments) < self.num_segments:
            for break_point in ['. ', '! ', '? ', '; ', ', ']:
                split_at = remaining.rfind(break_point, 0, target_length + 50)
                if split_at != -1:
                    segments.append(remaining[:split_at+1].strip())
                    remaining = remaining[split_at+1:].strip()
                    break
            else:
                last_space = remaining.rfind(' ', 0, target_length)
                if last_space != -1:
                    segments.append(remaining[:last_space].strip())
                    remaining = remaining[last_space:].strip()
                else:
                    segments.append(remaining[:target_length].strip())
                    remaining = remaining[target_length:].strip()
        
        if remaining and segments:
            segments[-1] += " " + remaining
            
        return text, segments[:self.num_segments]

    def _create_image_prompts(self, story, segments):
        """Generate consistent visual prompts"""
        prompt = f"""IMAGE PROMPT GUIDELINES:
            1. FIRST IDENTIFY ALL CHARACTERS:
            Read this story and list ALL characters with their visual features:
            {story}

            2. FOR EACH SEGMENT:
             Create image prompts for each of the {len(segments)} segments:
                {json.dumps(segments)}

                Create image prompts following these rules:
                - USE CHARACTER TITLES: "Young girl with red braids" not "she"
                - ONLY SHOW CHARACTERS EXPLICITLY MENTIONED IN THE SEGMENT
                - MAINTAIN IDENTICAL FEATURES WHEN SAME CHARACTER REAPPEARS
                - DO NOT USE NAMES OR PRONOUNS instead use descriptive terms like "young girl with red braids", "old man with white beard", "little boy with glasses", "little fox with big ears", "little cute rabbit" etc.
                - ART STYLE: Digital art cartoonish (consistent for all)
                - MAX 120 CHARACTERS PER PROMPT

                EXAMPLE PROMPTS:
                1. "Cheerful baker in striped apron kneading dough, cartoonish digital art"
                2. "Village market with colorful stalls and fresh bread, digital art"
                3. "Same baker decorating cake with strawberries, cartoon style"

            3. TECHNICAL:
            - Max 120 characters per prompt
            - Same art style for all prompts
            - No pronouns - use descriptive terms
            - Ensure visual continuity between related segments"""
        
        response = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": "You are a consistent visual designer that maintains character continuity. You Design prompts for Midjourney."},
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
        segment_durations = []
        
        for i, text in enumerate(segments):
            path = temp_path / f"audio_{i}.mp3"
            communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
            await communicate.save(str(path))
            audio_files.append(str(path))
            
            # Get duration of segment
            audio = AudioFileClip(str(path))
            segment_durations.append(audio.duration)
            audio.close()
        
        # Combine audio files
        combined = concatenate_audioclips([AudioFileClip(f) for f in audio_files])
        final_path = temp_path / "final_audio.mp3"
        combined.write_audiofile(str(final_path))
        combined.close()
        
        return final_path, segment_durations

    async def _compile_video(self, images, segments, audio_path, temp_path):
        """Create final video with memory optimizations"""
        try:
            audio_path, segment_durations = audio_path
            clips = []
            
            for image_array, segment, duration in zip(images, segments, segment_durations):
                # Calculate words per second for this segment
                words = segment.split()
                words_per_second = len(words) / duration
                
                # Split text into chunks that match speaking rhythm
                chunk_size = max(1, int(words_per_second * 1.5))  # 1.5-second chunks
                chunks = []
                current_chunk = []
                
                for word in words:
                    current_chunk.append(word)
                    if len(current_chunk) >= chunk_size or word[-1] in '.!?':
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Create sub-clips for each text chunk
                chunk_duration = duration / len(chunks)
                for i, chunk in enumerate(chunks):
                    image_with_caption = self._add_captions(image_array, chunk)
                    sub_clip = ImageClip(image_with_caption).set_duration(chunk_duration)
                    clips.append(sub_clip)
            
            # Combine all clips
            final_clip = concatenate_videoclips(clips)
            final_clip = final_clip.set_audio(AudioFileClip(str(audio_path)))
            
            # Optimized video writing parameters
            temp_video_path = temp_path / "temp_video.mp4"
            final_clip.write_videofile(
                str(temp_video_path),
                # fps=24,
                # threads=4,
                fps=15,  # Reduced from 24
                threads=2,  # Reduced from 4
                preset='ultrafast',
                # ffmpeg_params=['-movflags', '+faststart'],
                ffmpeg_params=[
                    '-movflags', '+faststart',
                    '-vf', 'scale=720:1280',  # Force scale
                    '-c:v', 'libx264',  # Hardware-friendly codec
                    '-crf', '28'  # Higher compression
                ],
                logger=None
            )
            
            # Clear memory-intensive objects early
            del images
            del clips
            del final_clip
            
            # Read file into memory
            with open(temp_video_path, 'rb') as f:
                video_bytes = BytesIO(f.read())
            
            video_bytes.seek(0)
            return video_bytes
            
        except Exception as e:
            raise RuntimeError(f"Video compilation failed: {str(e)}")
        # finally:
        #     if 'final_clip' in locals():
        #         final_clip.close()

    def _add_captions(self, image_array, text):
        """Add captions with proper placement"""
        img = Image.fromarray(image_array)
        draw = ImageDraw.Draw(img)

        # Attempt to load a font, fallback to default if not found
        # try:
        #     font = ImageFont.truetype("arialbd.ttf", size=80)
        # except:
        #     try:
        #         font = ImageFont.truetype("Arial_Bold.ttf", size=80)
        #     except:
        #         font = ImageFont.load_default()  # Fallback to default font
        
        # Dynamic font size based on video height (6% of screen height)
        base_font_size = int(self.height * 0.04)  # ~77px for 1280 height
        dynamic_font = None
        
        # Try different font options with descending priority
        font_paths = [
            "arialbd.ttf", 
            "Arial_Bold.ttf",
            "DejaVuSans-Bold.ttf",  # Common Linux font
            "LiberationSans-Bold.ttf"  # Another common fallback
        ]
        
        for path in font_paths:
            try:
                dynamic_font = ImageFont.truetype(path, size=base_font_size)
                break
            except:
                continue
                
        if not dynamic_font:  # Ultimate fallback with scaling
            dynamic_font = ImageFont.load_default()
            # Scale up default font using 2x transform
            dynamic_font = dynamic_font.font_variant(size=base_font_size*2)
        
        # Split into shorter lines for better readability
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= 4:  # Limit to 4 words per line
                lines.append(' '.join(current_line))
                current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate text block positioning
        # line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        line_height = dynamic_font.getbbox("A")[3] - dynamic_font.getbbox("A")[1]
        total_height = len(lines) * (line_height + 10)
        y_position = (self.height // 2) + ((self.height // 2 - total_height) // 2) # Place near bottom with padding
        
        # Ensure y_position is within bounds
        if y_position < 0:
            y_position = 10  # Adjust if it goes off-screen
        
        # Draw each line
        for line in lines:
            # bbox = font.getbbox(line)
            bbox = dynamic_font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            x_position = max(100, (self.width - text_width) // 2)
            
            # Draw outline for better visibility
            draw.text(
                (x_position, y_position),
                line,
                # font=font,
                font=dynamic_font,
                fill="#FFFF00",  # Text fill color
                stroke_width=3,  # Outline width
                stroke_fill="#000000"  # Outline color
            )
            y_position += line_height + 10  # Move down for the next line
        
        return np.array(img)