# # Core audio processing
# pyaudio>=0.2.11
# numpy>=1.21.0

# # Graphics and visualization
# pygame>=2.1.0

# # Audio analysis (optional but recommended)
# librosa>=0.9.0
# scipy>=1.7.0

# # Additional audio processing utilities
# soundfile>=0.10.0

# # Performance optimization
# numba>=0.56.0  # For JIT compilation of audio processing functions

# # Cross-platform compatibility
# setuptools>=60.0.0
# wheel>=0.37.0


#!/usr/bin/env python3
"""
Real-Time Audio Visualizer Application
A high-quality, cross-platform audio visualizer with dynamic color adaptation
and multiple visualization modes.
"""

import sys
import numpy as np
import pyaudio
import threading
import time
import colorsys
from collections import deque
import argparse

# Graphics and UI imports
try:
    import pygame
    GRAPHICS_BACKEND = 'pygame'
except ImportError:
    try:
        import tkinter as tk
        from tkinter import ttk
        GRAPHICS_BACKEND = 'tkinter'
    except ImportError:
        print("Error: No graphics backend available. Please install pygame.")
        sys.exit(1)

# Audio processing imports
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available. Some audio analysis features disabled.")


class AudioProcessor:
    """Handles real-time audio input and analysis."""
    
    def __init__(self, sample_rate=44100, chunk_size=4096, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_data = deque(maxlen=chunk_size * 4)
        self.frequency_data = np.zeros(chunk_size // 2)
        self.is_running = False
        self.audio_thread = None
        
        # Audio analysis parameters
        self.bass_range = (20, 250)
        self.mid_range = (250, 2000)
        self.treble_range = (2000, 20000)
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Audio characteristics
        self.bass_intensity = 0
        self.mid_intensity = 0
        self.treble_intensity = 0
        self.dominant_frequency = 0
        self.volume = 0
        self.tempo = 0
        
    def list_audio_devices(self):
        """List available audio input devices."""
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append((i, info['name']))
        return devices
    
    def start_recording(self, device_index=None):
        """Start audio recording from specified device."""
        try:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            
            # Start analysis thread
            self.audio_thread = threading.Thread(target=self._analysis_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio data."""
        audio_array = np.frombuffer(in_data, dtype=np.float32)
        self.audio_data.extend(audio_array)
        return (None, pyaudio.paContinue)
    
    def _analysis_loop(self):
        """Background thread for audio analysis."""
        while self.is_running:
            if len(self.audio_data) >= self.chunk_size:
                # Get recent audio data
                recent_data = np.array(list(self.audio_data)[-self.chunk_size:])
                
                # Perform FFT
                fft_data = np.fft.fft(recent_data)
                self.frequency_data = np.abs(fft_data[:len(fft_data)//2])
                
                # Analyze audio characteristics
                self._analyze_audio_characteristics()
                
            time.sleep(0.01)  # 100 FPS analysis
    
    def _analyze_audio_characteristics(self):
        """Analyze bass, mid, treble, volume, and other characteristics."""
        if len(self.frequency_data) == 0:
            return
            
        # Calculate frequency bins
        freqs = np.fft.fftfreq(len(self.frequency_data) * 2, 1/self.sample_rate)[:len(self.frequency_data)]
        
        # Split into frequency ranges
        bass_mask = (freqs >= self.bass_range[0]) & (freqs <= self.bass_range[1])
        mid_mask = (freqs >= self.mid_range[0]) & (freqs <= self.mid_range[1])
        treble_mask = (freqs >= self.treble_range[0]) & (freqs <= self.treble_range[1])
        
        # Calculate intensities
        self.bass_intensity = np.mean(self.frequency_data[bass_mask]) if np.any(bass_mask) else 0
        self.mid_intensity = np.mean(self.frequency_data[mid_mask]) if np.any(mid_mask) else 0
        self.treble_intensity = np.mean(self.frequency_data[treble_mask]) if np.any(treble_mask) else 0
        
        # Overall volume
        self.volume = np.mean(self.frequency_data)
        
        # Dominant frequency
        if len(freqs) > 0:
            dominant_idx = np.argmax(self.frequency_data)
            self.dominant_frequency = freqs[dominant_idx] if dominant_idx < len(freqs) else 0
    
    def stop_recording(self):
        """Stop audio recording and analysis."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup PyAudio resources."""
        self.stop_recording()
        if hasattr(self, 'pa'):
            self.pa.terminate()


class ColorManager:
    """Manages dynamic color adaptation based on audio characteristics."""
    
    def __init__(self):
        self.base_hue = 0
        self.saturation = 0.8
        self.brightness = 0.6
        self.color_history = deque(maxlen=60)  # 1 second at 60 FPS
        
    def update_colors(self, bass, mid, treble, volume, dominant_freq):
        """Update colors based on audio characteristics."""
        # Map frequency ranges to color characteristics
        # Bass -> Red/Orange (0-60 degrees)
        # Mid -> Green/Yellow (60-180 degrees) 
        # Treble -> Blue/Purple (180-300 degrees)
        
        # Calculate hue based on dominant frequencies
        bass_weight = bass / (bass + mid + treble + 0.001)
        mid_weight = mid / (bass + mid + treble + 0.001)
        treble_weight = treble / (bass + mid + treble + 0.001)
        
        target_hue = (bass_weight * 30 + mid_weight * 120 + treble_weight * 240) % 360
        
        # Smooth hue transitions
        self.base_hue += (target_hue - self.base_hue) * 0.1
        self.base_hue = self.base_hue % 360
        
        # Adjust saturation based on volume
        self.saturation = max(0.3, min(1.0, 0.5 + volume * 0.5))
        
        # Adjust brightness based on overall intensity
        total_intensity = bass + mid + treble
        self.brightness = max(0.3, min(0.9, 0.4 + total_intensity * 0.1))
        
        # Store in history for effects
        self.color_history.append({
            'hue': self.base_hue,
            'saturation': self.saturation,
            'brightness': self.brightness,
            'bass': bass,
            'mid': mid,
            'treble': treble
        })
    
    def get_primary_color(self):
        """Get the primary color as RGB tuple (0-255)."""
        rgb = colorsys.hsv_to_rgb(self.base_hue/360, self.saturation, self.brightness)
        return tuple(int(c * 255) for c in rgb)
    
    def get_complementary_color(self):
        """Get complementary color."""
        comp_hue = (self.base_hue + 180) % 360
        rgb = colorsys.hsv_to_rgb(comp_hue/360, self.saturation, self.brightness)
        return tuple(int(c * 255) for c in rgb)
    
    def get_spectrum_colors(self, num_colors):
        """Get an array of colors for spectrum visualization."""
        colors = []
        for i in range(num_colors):
            hue = (self.base_hue + (i / num_colors) * 120) % 360
            brightness = self.brightness * (0.7 + 0.3 * (i / num_colors))
            rgb = colorsys.hsv_to_rgb(hue/360, self.saturation, brightness)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors


class PygameVisualizer:
    """Pygame-based visualizer with multiple visualization modes."""
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Real-Time Audio Visualizer")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.fullscreen = False
        
        # Visualization modes
        self.modes = ['spectrum', 'circular', 'waveform', 'particles', 'dna']
        self.current_mode = 0
        
        # Visual parameters
        self.sensitivity = 1.0
        self.smoothing = 0.8
        self.trail_length = 0.1
        
        # Particle system
        self.particles = []
        self.init_particles()
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
    def init_particles(self):
        """Initialize particle system."""
        self.particles = []
        for _ in range(300):
            particle = {
                'x': np.random.random() * self.width,
                'y': np.random.random() * self.height,
                'vx': (np.random.random() - 0.5) * 4,
                'vy': (np.random.random() - 0.5) * 4,
                'size': np.random.random() * 3 + 1,
                'life': np.random.random(),
                'color': [255, 255, 255]
            }
            self.particles.append(particle)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f or event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    self.current_mode = event.key - pygame.K_1
                elif event.key == pygame.K_UP:
                    self.sensitivity = min(3.0, self.sensitivity + 0.1)
                elif event.key == pygame.K_DOWN:
                    self.sensitivity = max(0.1, self.sensitivity - 0.1)
            
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.init_particles()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
            self.width, self.height = 1200, 800
        self.init_particles()
    
    def draw_spectrum(self, frequency_data, colors):
        """Draw spectrum bars visualization."""
        if len(frequency_data) == 0:
            return
            
        bar_width = self.width / len(frequency_data)
        
        for i, amplitude in enumerate(frequency_data[:len(frequency_data)//4]):  # Show first quarter
            height = int(amplitude * self.sensitivity * self.height * 0.8)
            x = i * bar_width * 4
            y = self.height - height
            
            color_idx = int(i / len(frequency_data) * len(colors)) % len(colors)
            color = colors[color_idx]
            
            # Draw bar with gradient effect
            for j in range(0, height, 5):
                alpha = max(50, 255 - j)
                bar_color = (*color, alpha)
                rect = pygame.Rect(x, y + j, bar_width * 4, 5)
                
                # Create surface with per-pixel alpha
                surf = pygame.Surface((bar_width * 4, 5), pygame.SRCALPHA)
                surf.fill(bar_color)
                self.screen.blit(surf, rect)
    
    def draw_circular_spectrum(self, frequency_data, color_manager):
        """Draw circular spectrum visualization."""
        if len(frequency_data) == 0:
            return
            
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) // 4
        
        num_bars = min(128, len(frequency_data))
        angle_step = 2 * np.pi / num_bars
        
        for i in range(num_bars):
            angle = i * angle_step
            amplitude = frequency_data[i] * self.sensitivity
            bar_length = amplitude * radius * 0.8
            
            # Calculate positions
            inner_x = center_x + np.cos(angle) * radius
            inner_y = center_y + np.sin(angle) * radius
            outer_x = center_x + np.cos(angle) * (radius + bar_length)
            outer_y = center_y + np.sin(angle) * (radius + bar_length)
            
            # Color based on position
            hue = (color_manager.base_hue + (i / num_bars) * 360) % 360
            rgb = colorsys.hsv_to_rgb(hue/360, color_manager.saturation, color_manager.brightness)
            color = tuple(int(c * 255) for c in rgb)
            
            # Draw line with thickness based on amplitude
            thickness = max(1, int(amplitude * 5))
            pygame.draw.line(self.screen, color, (inner_x, inner_y), (outer_x, outer_y), thickness)
    
    def draw_waveform(self, audio_data, color_manager):
        """Draw waveform visualization."""
        if len(audio_data) == 0:
            return
            
        # Downsample for display
        step = max(1, len(audio_data) // self.width)
        samples = audio_data[::step]
        
        if len(samples) < 2:
            return
            
        points = []
        for i, sample in enumerate(samples):
            x = (i / len(samples)) * self.width
            y = self.height // 2 + sample * self.sensitivity * self.height * 0.3
            points.append((x, y))
        
        if len(points) > 1:
            color = color_manager.get_primary_color()
            pygame.draw.lines(self.screen, color, False, points, 3)
            
            # Add glow effect
            glow_color = (*color, 100)
            for offset in range(1, 6):
                glow_points = [(x, y + offset) for x, y in points]
                pygame.draw.lines(self.screen, glow_color, False, glow_points, 1)
    
    def draw_particles(self, audio_data, color_manager):
        """Draw particle system visualization."""
        # Update particles based on audio
        audio_influence = color_manager.brightness * 10
        
        for particle in self.particles:
            # Move particle
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Audio influence
            particle['vx'] += (np.random.random() - 0.5) * audio_influence
            particle['vy'] += (np.random.random() - 0.5) * audio_influence
            
            # Damping
            particle['vx'] *= 0.99
            particle['vy'] *= 0.99
            
            # Wrap around screen
            if particle['x'] < 0:
                particle['x'] = self.width
            elif particle['x'] > self.width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = self.height
            elif particle['y'] > self.height:
                particle['y'] = 0
            
            # Update color
            hue = (color_manager.base_hue + particle['life'] * 60) % 360
            rgb = colorsys.hsv_to_rgb(hue/360, color_manager.saturation, color_manager.brightness)
            particle['color'] = [int(c * 255) for c in rgb]
            
            # Draw particle
            size = int(particle['size'] * (1 + audio_influence * 0.1))
            pygame.draw.circle(self.screen, particle['color'], 
                             (int(particle['x']), int(particle['y'])), size)
    
    def draw_dna_helix(self, frequency_data, color_manager):
        """Draw DNA helix visualization."""
        if len(frequency_data) == 0:
            return
            
        center_x = self.width // 2
        amplitude = 100 + color_manager.bass_intensity * 2
        frequency = 0.01 + color_manager.mid_intensity * 0.0001
        
        # Draw two helixes
        for strand in range(2):
            points = []
            for y in range(0, self.height, 4):
                phase = strand * np.pi + color_manager.base_hue * 0.01
                x = center_x + np.sin(y * frequency + phase) * amplitude
                points.append((x, y))
            
            if len(points) > 1:
                hue = (color_manager.base_hue + strand * 180) % 360
                rgb = colorsys.hsv_to_rgb(hue/360, color_manager.saturation, color_manager.brightness)
                color = tuple(int(c * 255) for c in rgb)
                
                pygame.draw.lines(self.screen, color, False, points, 4)
        
        # Draw connecting lines
        connection_color = color_manager.get_complementary_color()
        for y in range(0, self.height, 30):
            phase1 = color_manager.base_hue * 0.01
            phase2 = np.pi + color_manager.base_hue * 0.01
            x1 = center_x + np.sin(y * frequency + phase1) * amplitude
            x2 = center_x + np.sin(y * frequency + phase2) * amplitude
            
            pygame.draw.line(self.screen, connection_color, (x1, y), (x2, y), 2)
    
    def draw_ui(self, audio_processor, fps):
        """Draw user interface elements."""
        # Semi-transparent background for UI
        ui_surface = pygame.Surface((300, 200), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 128))
        self.screen.blit(ui_surface, (10, 10))
        
        # Text information
        y_offset = 20
        texts = [
            f"Mode: {self.modes[self.current_mode].title()} ({self.current_mode + 1})",
            f"FPS: {fps:.1f}",
            f"Volume: {audio_processor.volume:.2f}",
            f"Bass: {audio_processor.bass_intensity:.2f}",
            f"Mid: {audio_processor.mid_intensity:.2f}",
            f"Treble: {audio_processor.treble_intensity:.2f}",
            f"Dominant Freq: {audio_processor.dominant_frequency:.0f} Hz",
            f"Sensitivity: {self.sensitivity:.1f}",
            "",
            "Controls:",
            "1-5: Change mode",
            "F/F11: Fullscreen",
            "↑/↓: Sensitivity",
            "ESC: Exit"
        ]
        
        for text in texts:
            if text:
                surface = self.small_font.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (20, y_offset))
            y_offset += 18
    
    def render(self, audio_processor, color_manager):
        """Main rendering function."""
        # Clear screen with trail effect
        trail_surface = pygame.Surface((self.width, self.height))
        trail_surface.set_alpha(int(255 * self.trail_length))
        trail_surface.fill((0, 0, 0))
        self.screen.blit(trail_surface, (0, 0))
        
        # Get audio data
        frequency_data = audio_processor.frequency_data
        audio_data = list(audio_processor.audio_data) if audio_processor.audio_data else []
        
        # Draw visualization based on current mode
        colors = color_manager.get_spectrum_colors(len(frequency_data) if len(frequency_data) > 0 else 256)
        
        if self.modes[self.current_mode] == 'spectrum':
            self.draw_spectrum(frequency_data, colors)
        elif self.modes[self.current_mode] == 'circular':
            self.draw_circular_spectrum(frequency_data, color_manager)
        elif self.modes[self.current_mode] == 'waveform':
            self.draw_waveform(audio_data, color_manager)
        elif self.modes[self.current_mode] == 'particles':
            self.draw_particles(audio_data, color_manager)
        elif self.modes[self.current_mode] == 'dna':
            self.draw_dna_helix(frequency_data, color_manager)
        
        # Draw UI
        fps = self.clock.get_fps()
        self.draw_ui(audio_processor, fps)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Real-Time Audio Visualizer')
    parser.add_argument('--list-devices', action='store_true', 
                       help='List available audio input devices')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio input device index')
    parser.add_argument('--width', type=int, default=1200,
                       help='Window width')
    parser.add_argument('--height', type=int, default=800,
                       help='Window height')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Start in fullscreen mode')
    
    args = parser.parse_args()
    
    # Initialize components
    audio_processor = AudioProcessor()
    
    # List devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        devices = audio_processor.list_audio_devices()
        for idx, name in devices:
            print(f"  {idx}: {name}")
        return
    
    # Initialize visualizer
    if GRAPHICS_BACKEND == 'pygame':
        visualizer = PygameVisualizer(args.width, args.height)
        if args.fullscreen:
            visualizer.toggle_fullscreen()
    else:
        print("Error: Pygame not available. Please install pygame for graphics.")
        return
    
    # Initialize color manager
    color_manager = ColorManager()
    
    # Start audio recording
    print("Starting audio recording...")
    if not audio_processor.start_recording(args.device):
        print("Failed to start audio recording. Continuing with demo mode...")
    
    try:
        print("Audio Visualizer started!")
        print("Controls:")
        print("  1-5: Change visualization mode")
        print("  F/F11: Toggle fullscreen")
        print("  ↑/↓: Adjust sensitivity")
        print("  ESC: Exit")
        
        # Main loop
        while visualizer.running:
            # Handle events
            visualizer.handle_events()
            
            # Update colors based on audio
            color_manager.update_colors(
                audio_processor.bass_intensity,
                audio_processor.mid_intensity,
                audio_processor.treble_intensity,
                audio_processor.volume,
                audio_processor.dominant_frequency
            )
            
            # Render frame
            visualizer.render(audio_processor, color_manager)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Cleanup
        audio_processor.stop_recording()
        if GRAPHICS_BACKEND == 'pygame':
            pygame.quit()
    
    print("Audio Visualizer stopped.")


if __name__ == "__main__":
    main()