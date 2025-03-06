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
from tempfile import TemporaryDirectory, gettempdir
import io
from task_queue import async_task

class VideoGenerator:
    def __init__(self):
        self.width = 720
        self.height = 1280
        self.target_duration = 45  # Changed from 30 to 45
        self.max_segment_duration = 4  # Changed from 3 to 4
        self.num_segments = math.ceil(self.target_duration / self.max_segment_duration)
        self.speech_rate = "-10%"
        self.model = "mistral-large"
        # Create output directory
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)

    def download_image(self, prompt, seed, max_retries=3):
        """Download image from pollinations.ai with proper error handling"""
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://pollinations.ai/p/{encoded_prompt}?width={self.width}&height={self.height}&seed={seed}&model=flux&nologo=true"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download image after {max_retries} attempts: {str(e)}")
                time.sleep(2)
                continue

    @async_task
    async def _generate_video_task(self, topic, progress_callback=None, task_id=None):
        if hasattr(self, 'last_video'):
            del self.last_video
        import gc; gc.collect()
        
        with TemporaryDirectory(dir=gettempdir()) as temp_dir:
            temp_path = Path(temp_dir)
            try:
                if progress_callback:
                    await progress_callback("📝 Crafting your story...")
                story, segments = await asyncio.to_thread(self._generate_story, topic)
                
                if progress_callback:
                    await progress_callback("🎨 Creating visuals...")
                image_prompts = self._create_image_prompts(story, segments)
                images = []
                for i, prompt in enumerate(image_prompts):
                    for attempt in range(3):
                        try:
                            seed = random.randint(1, 999999)
                            image = await asyncio.to_thread(self.download_image, prompt, seed)
                            images.append(np.array(image))
                            del image
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise RuntimeError(f"Failed to generate image {i+1}: {str(e)}")
                            await asyncio.sleep(1)
                
                if progress_callback:
                    await progress_callback("🔊 Recording narration...")
                audio_result = await self._generate_audio(segments, temp_path)
                
                if progress_callback:
                    await progress_callback("🎥 Compiling final video...")
                # Offload the compilation to a separate thread
                video_bytes = await asyncio.to_thread(self._compile_video, images, segments, audio_result, temp_path)
                
                # Return the video bytes directly (no persistent saving)
                return video_bytes.getvalue()
            except Exception as e:
                print(f"Video generation error: {str(e)}")
                raise RuntimeError(f"Video generation failed: {str(e)}")
            finally:
                # Cleanup temporary files in temp_path
                try:
                    for f in temp_path.glob("*"):
                        f.unlink()
                except Exception:
                    pass

    async def generate_video(self, topic, progress_callback=None):
        """Start async video generation and return task ID"""
        task_id = f"video_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Queue the task and return ID immediately
        await self._generate_video_task(task_id, topic, progress_callback)
        return task_id

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
        # Clean and normalize text first
        text = text.strip()
        if not text:
            return [], ""
            
        segments = []
        story = text
        target_length = len(text) // self.num_segments
        
        # Split by sentences first
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + "."  # Add back the period
            if current_length + len(sentence) <= target_length or not current_segment:
                current_segment.append(sentence)
                current_length += len(sentence)
            else:
                segments.append(" ".join(current_segment))
                current_segment = [sentence]
                current_length = len(sentence)
        
        # Add any remaining segment
        if current_segment:
            segments.append(" ".join(current_segment))
        
        # If we need more segments, split the longest ones
        while len(segments) < self.num_segments and any(len(s) > target_length for s in segments):
            # Find longest segment
            longest_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
            long_segment = segments[longest_idx]
            
            # Split at the nearest sentence or space
            split_point = long_segment.rfind('.', 0, len(long_segment)//2)
            if split_point == -1:
                split_point = long_segment.rfind(' ', 0, len(long_segment)//2)
            
            if split_point != -1:
                segments[longest_idx:longest_idx+1] = [
                    long_segment[:split_point].strip(),
                    long_segment[split_point:].strip()
                ]
        
        # Ensure we have exactly the number of segments needed
        while len(segments) > self.num_segments:
            # Merge shortest adjacent segments
            lengths = [len(s) for s in segments]
            min_combined_idx = min(range(len(segments)-1),
                                 key=lambda i: lengths[i] + lengths[i+1])
            segments[min_combined_idx:min_combined_idx+2] = [
                segments[min_combined_idx] + " " + segments[min_combined_idx+1]
            ]
        
        while len(segments) < self.num_segments:
            # Split longest segment
            max_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
            segment = segments[max_idx]
            split_point = len(segment) // 2
            split_point = segment.rfind(' ', 0, split_point)
            if split_point == -1:
                split_point = len(segment) // 2
            
            segments[max_idx:max_idx+1] = [
                segment[:split_point].strip(),
                segment[split_point:].strip()
            ]
        
        return story, segments

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
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
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

    def _compile_video(self, images, segments, audio_path, temp_path):
        """Create final video with improved temporary file handling"""
        os.environ["FFMPEG_TEMP_DIR"] = str(temp_path)
        
        try:
            audio_path, segment_durations = audio_path
            clips = []
            
            for image_array, segment, duration in zip(images, segments, segment_durations):
                # Calculate words per second for this segment
                words = segment.split()
                words_per_second = len(words) / duration
                
                # Split text into chunks that match speaking rhythm
                chunk_size = max(1, int(words_per_second * 1.5))
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
                for chunk in chunks:
                    image_with_caption = self._add_captions(image_array, chunk)
                    sub_clip = ImageClip(image_with_caption).set_duration(chunk_duration)
                    clips.append(sub_clip)
            
            # Combine clips and add audio
            final_clip = concatenate_videoclips(clips)
            final_clip = final_clip.set_audio(AudioFileClip(str(audio_path)))
            
            # Create output path and ensure temp directory exists
            output_path = temp_path / "output.mp4"
            temp_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(temp_path), 0o777)
            
            # Create temp audio file path
            temp_audio = temp_path / "temp_audio.mp3"
            
            final_clip.write_videofile(
                str(output_path),
                fps=15,
                threads=2,
                preset='ultrafast',
                temp_audiofile=str(temp_audio),
                ffmpeg_params=[
                    '-movflags', '+faststart',
                    '-vf', f'scale={self.width}:{self.height}',
                    '-c:v', 'libx264',
                    '-crf', '28',
                    '-tune', 'fastdecode'
                ],
                logger=None
            )
            
            # Read file into memory
            video_bytes = BytesIO()
            with open(output_path, 'rb') as f:
                video_bytes.write(f.read())
            
            video_bytes.seek(0)
            return video_bytes
            
        except Exception as e:
            raise RuntimeError(f"Video compilation failed: {str(e)}")
        finally:
            # Clean up
            try:
                if 'final_clip' in locals():
                    final_clip.close()
                for clip in clips:
                    clip.close()
            except:
                pass

    def _add_captions(self, image_array, text):
        """Universal font size solution"""
        img = Image.fromarray(image_array)
        draw = ImageDraw.Draw(img)

        # Dynamic font size based on video height (6% of screen height)
        base_font_size = int(self.height * 0.04)  # ~77px for 1280 height
        dynamic_font = None
        
        # Try different font options with descending priority
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Common on Linux
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "arialbd.ttf", 
            "Arial_Bold.ttf",
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
            dynamic_font = dynamic_font.font_variant(size=base_font_size)
        
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
        line_height = dynamic_font.getbbox("A")[3] - dynamic_font.getbbox("A")[1]
        total_height = len(lines) * (line_height + 10)
        y_position = (self.height // 2) + ((self.height // 2 - total_height) // 2) # Place near bottom with padding
        
        # Ensure y_position is within bounds
        if y_position < 0:
            y_position = 10  # Adjust if it goes off-screen
        
        # Draw each line
        for line in lines:
            bbox = dynamic_font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            x_position = max(100, (self.width - text_width) // 2)
            
            # Draw outline for better visibility
            draw.text(
                (x_position, y_position),
                line,
                font=dynamic_font,
                fill="#FFFF00",  # Text fill color
                stroke_width=3,  # Outline width
                stroke_fill="#000000"  # Outline color
            )
            y_position += line_height + 10  # Move down for the next line
        
        return np.array(img)