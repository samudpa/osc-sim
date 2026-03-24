import numpy as np
import cupy as cp
import soundfile as sf
import subprocess
import time
import math
import os
from numba import cuda
from cupyx.scipy.signal import fftconvolve
from cupyx.scipy.ndimage import zoom, gaussian_filter

@cuda.jit
def _draw_beam_cuda(result_image, x, y, intensity):
    """
    (GPU) CUDA kernel that draws a curve with an intensity on 2D image at coordinates x, y.
    Accelerated with Numba CUDA. 1 thread = 1 point
    """

    # get thread index
    idx = cuda.grid(1)

    # ignore indexes over the number of audio samples
    if idx < x.shape[0]:

        xi = x[idx]
        yi = y[idx]
        zi = intensity[idx]

        max_y = result_image.shape[0]
        max_x = result_image.shape[1]

        # check bounds and optionally skip
        if 0 <= yi < max_y and 0 <= xi < max_x:
            
            # !! use atomic addition
            # so if beam crosses over itself, the result is the sum of all crossings
            cuda.atomic.add(result_image, (yi, xi), zi)

class AnalogOsc:

    def __init__(self, filename,
        width=1080, height=1080, fps=60, shutter=0.5,                       # camera
        subsampling=1000000, upscaling=2, scale=0.9,                        # simulation quality
        decay_time=0.015, beam_power=7e6, flash_factor=1.5,                 # physics
        line_width=1, glow_radius=15, glow_downscale_factor=2,              # blur kernel
        jitter=1e-3, jitter_corr=1/60, grain=0.08, background_level=0.,     # synthetic noise
        grid_opacity=0.99, grid_params=None,                                # graticule
        dual_color=False, xy_mode=True, rotate_scope=False, flip_y=False):  # rotation and polarity

        # parameters
        self.audio_filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.shutter = shutter # shutter open period as a fraction of frametime
        self.subsampling = subsampling # how many points, per frame, the beam is interpolated to
        self.upscaling = upscaling # antialiasing upscale factor
        self.scale = scale # scale the signal by this factor
        self.decay_time = decay_time # phosphor decay time in seconds
        self.background_level = background_level # background signal level (1 means fully lit)

        # beam parameters
        self.glow_radius = glow_radius # blur scale radius in pixels (small and wide exponential glow/halo)
        self.line_width = line_width # line width in pixels (small gaussian center)
        self.beam_power = beam_power # energy per second of the electron beam
        self.flash_factor = flash_factor # multiplying factor for the fluorescence "flash"

        # geometry parameters
        self.xy_mode = xy_mode # True for vectorscope X-Y display, False for traditional t-Y display
        self.rotate_scope = rotate_scope # if True, rotates X-Y display by 45°
        self.flip_y = flip_y # if True, flips display vertically

        # noise parameters (jitter and grain)
        self.jitter = jitter # synthetic analog voltage jitter on X, Y
        self.jitter_corr = jitter_corr # auto-correlation window for the jitter, in seconds
        self.grain = grain # multiplicative grain variance

        # if False, it will speed up drawing by drawing the combined glow+flash camera output in one pass
        # this is not possible if the colors need to be different.
        # this boolean will get automatically changed by .render() if you pass separate colors there
        self.dual_color = dual_color

        # calculate derived parameters
        self.samplerate = fps * subsampling
        self.frame_time = 1.0 / fps
        self.shutter_time = self.frame_time * shutter
        self.shutter_steps = int(self.shutter_time * self.samplerate)
        self.decay_factor = np.exp(-self.frame_time/decay_time)
        self.phos_point_power = beam_power / self.samplerate # point power for the phosphorescence contribution
        self.fluo_point_power = flash_factor * beam_power / self.samplerate # point power for the fluorescence contribution

        # initialize intensity-curves-over-time
        # - state_curve: used in the calculation of the underlying physical state
        # - video_glow_curve: used in the "glow" (phosphorescence) calculation for the camera output
        # - video_flash_curve: used in the "flash" (fluorescence) calculation for the camera output
        # - video_combined_curve: used in case base color and flash color are identical
        self.state_curve, self.video_glow_curve, self.video_flash_curve = self._build_intensity_curves()
        self.video_combined_curve = self.video_glow_curve + self.video_flash_curve

        # initialize the blur kernel
        self.glow_downscale_factor = glow_downscale_factor
        self.glow_kernel = self._build_glow_kernel(glow_radius*upscaling/glow_downscale_factor)

        # initialize the graticule/grid
        self.graticule = (grid_opacity > 0)
        if self.graticule:
            if grid_params is None: grid_params = {}
            default_grid_params = {
                'thickness': 1.5,
                'subdiv_length': 12,
                'scale': 0.97,
                'divs': 10,
                'subdivs': 5,
            }
            grid_params = default_grid_params | grid_params
            self.graticule_mask = self._build_graticule(grid_opacity, **grid_params)

        # load audio file
        self.x, self.y, self.audio_samplerate, audio_duration = self._load_audio(filename)
        self.total_samples = int(audio_duration * self.samplerate)
        self.duration = self.total_samples / self.samplerate

        # initialize phosphor state
        self._init_scope()

    def _hex_to_bgr(self, hex_color):
        """
        Converts a HEX color string to normalized BGR numpy array for OpenCV.
        """

        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        # push color to GPU
        return cp.array([b, g, r], dtype=cp.float32)

    def _build_intensity_curves(self):
        """
        (GPU) Initialize state frame and video frame intensity curves
        over entire frame-time and shutter-time, respectively.
        """

        # define state intensity multiplier over the entire frame
        t = cp.linspace(-self.frame_time, 0, self.subsampling, dtype=cp.float32)
        state_curve = self.phos_point_power * cp.exp(t/self.decay_time)

        # define video frame intensity multiplier over the shutter time
        t = cp.linspace(-self.shutter_time, 0, self.shutter_steps, dtype=cp.float32)
        fluorescence = self.fluo_point_power + 0*t
        phosphorescence = self.phos_point_power * self.decay_time * (1 - cp.exp(t/self.decay_time))

        return state_curve, phosphorescence, fluorescence

    def _build_glow_kernel(self, radius):
        """
        Returns the glowing, exponential part of the blur kernel
        - F: phosphor grain scatter (short exp)
        - G: glass halation (long exp)
        """

        kernel_radius = int(np.ceil(14 * radius))
        y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
        r = np.sqrt(x**2 + y**2)

        # F: small exponential spread
        F = np.exp(-r / (0.15*radius))
        F /= F.sum()

        # G: wide exponential spread
        G = np.exp(-r / (1.3*radius))
        G /= G.sum()

        # combined kernel
        kernel = 2/3*F + 1/3*G
        kernel /= kernel.sum()

        # push kernel to GPU
        return cp.asarray(kernel, dtype=cp.float32)

    def _apply_blur(self, image):
        """
        (GPU) Apply the multi-resolution blur/glow. Beam blurring is processed at full resolution,
        while wide scale glow is processed at low resolution using fftconvolve.
        """

        downscale = self.glow_downscale_factor

        # E: gaussian electron beam
        beam = gaussian_filter(image, self.line_width*self.upscaling)

        # multiply by graticule if enabled
        if self.graticule:
            beam = beam * self.graticule_mask
            image = image * self.graticule_mask

        # F+G: small and large exponential spread
        if downscale == 1:

            # apply kernel without downscaling
            glow = fftconvolve(image, self.glow_kernel, mode='same')
            glow = cp.maximum(glow, 0.)

        else:

            # downscale
            h, w = image.shape
            h_down = h // downscale
            w_down = w // downscale
            image_down = image.reshape((h_down, downscale, w_down, downscale)).mean(axis=(1,3))

            # apply glow kernel
            glow_down = fftconvolve(image_down, self.glow_kernel, mode='same')
            glow_down = cp.maximum(glow_down, 0.)

            # upscale (bilinear)
            glow = zoom(glow_down, downscale, order=1)

        # combine sharp beam gaussian and glow exponentials
        # 70% and 30% energy contribution
        final_image = 0.7 * beam + 0.3 * glow

        return final_image

    def _build_graticule(self, opacity, thickness, divs, subdivs, subdiv_length, scale):
        """
        Build a multiplicative mask for the graticule. Thickness represents line width.
        """

        if opacity <= 0: return None

        height = self.height * self.upscaling
        width = self.width * self.upscaling
        thickness *= self.upscaling
        subdiv_length *= self.upscaling
        total_divisions = subdivs * divs

        # initialize mask (fully transparent)
        mask = cp.ones((height, width), cp.float32)

        # find main div boundaries
        step_y = (scale*height) / divs
        step_x = (scale*width) / divs
        min_x = int(max(width/2 - divs/2 * step_x - thickness, 0))
        max_x = int(min(width/2 + divs/2 * step_x + thickness, width-1))
        min_y = int(max(height/2 - divs/2 * step_y - thickness, 0))
        max_y = int(min(height/2 + divs/2 * step_y + thickness, height-1))

        # find subdivs boundaries
        substep_y = (scale*height) / total_divisions
        substep_x = (scale*width) / total_divisions
        submin_x = int(max(width/2 - subdiv_length, 0))
        submax_x = int(min(width/2 + subdiv_length, width-1))
        submin_y = int(max(height/2 - subdiv_length, 0))
        submax_y = int(min(height/2 + subdiv_length, height-1))

        # draw divisions
        for i in range(0, total_divisions+1):

            xpos = width/2 + (i - total_divisions/2) * substep_x
            ypos = height/2 + (i - total_divisions/2) * substep_y

            if i % subdivs == 0:
                # draw full line
                mask[min_y:max_y, int(xpos - thickness) : int(xpos + thickness)] = 0. # vertical lines
                mask[int(ypos - thickness) : int(ypos + thickness), min_x:max_x] = 0. # horizontal lines
            else:
                # draw small subdivision
                mask[submin_y:submax_y, int(xpos - thickness) : int(xpos + thickness)] = 0. # vertical lines
                mask[int(ypos - thickness) : int(ypos + thickness), submin_x:submax_x] = 0. # horizontal lines

        mask = gaussian_filter(mask, 0.8 * self.upscaling) # small blur
        mask = 1. - ((1. - mask) * opacity)

        return mask

    def _load_audio(self, filename):
        """
        Load WAVE file.
        """

        data, samplerate = sf.read(filename)
        duration = len(data)/samplerate

        x = data[:,0]
        y = data[:,1]

        return x, y, samplerate, duration

    def _map_to_pixels(self, x, y):
        """
        (GPU) Convert t-X-Y signal into pixel coordinates on the map,
        accounting for XY or tY mode selection.
        """

        width = self.width * self.upscaling
        height = self.height * self.upscaling
        scale_x = self.scale
        scale_y = self.scale

        if self.flip_y: scale_y = - scale_y

        if self.xy_mode:

            # X-Y (vectorscope) mode
            if self.rotate_scope:
                xpx = (1 + (x-y)/cp.sqrt(2)*scale_x) * width/2
                ypx = (1 - (x+y)/cp.sqrt(2)*scale_y) * height/2
            else:
                xpx = (1 + x*scale_x) * width/2
                ypx = (1 - y*scale_y) * height/2

        else:

            # t-Y (oscilloscope) mode
            signal = (x+y)/2 # average the signal
            t = cp.linspace(-1, 1, len(signal))
            xpx = (1 + t*scale_x) * width/2
            ypx = (1 + signal*scale_y) * height/2

        # use int32 since 32-bits is good for GPU
        # and don't use uint32 in case i want scale >1 so that coordinates can go negative
        return xpx.astype(cp.int32), ypx.astype(cp.int32)

    def _get_jitter_noise(self, corr):
        """
        (GPU) Returns two self.subsampling long arrays of autocorrelated noise.
        corr represents the correlation time for the noise
        """

        # This function generates a limited amounts of knots,
        # and then interpolates between them with a cubic spline.
        # This is faster than generatic a big amount of white noise
        # and blurring with an passing window!

        knots = math.ceil(self.frame_time / corr)
        x = cp.random.randn(knots, dtype=cp.float32)
        y = cp.random.randn(knots, dtype=cp.float32)

        # interpolate with cubic spline
        factor = self.subsampling / knots
        x = zoom(x, factor, order=3)
        y = zoom(y, factor, order=3)

        # trim to subsampling steps
        x = x[:self.subsampling]
        y = y[:self.subsampling]

        return x, y

    def _get_frame_data(self, frame):
        """
        Returns X, Y data for a specific frame while resampling to target samplerate using Splines. Resampling happens on GPU.
        """

        # get start and end time/indices of frame
        start_t = frame * self.frame_time
        stop_t = (frame + 1) * self.frame_time
        start_idx = int(start_t * self.audio_samplerate)
        stop_idx = int(stop_t * self.audio_samplerate)

        # add a small padding
        padding = 3
        pad_start_idx = max(0, start_idx - padding)
        pad_stop_idx = min(len(self.x), stop_idx + padding)

        # extract chunk of data and immediately push to GPU
        # build original and target time arrays
        x = cp.asarray(self.x[pad_start_idx:pad_stop_idx], dtype=cp.float32)
        y = cp.asarray(self.y[pad_start_idx:pad_stop_idx], dtype=cp.float32)
        
        # interpolate using cubic splines
        target_length = int((pad_stop_idx - pad_start_idx) / self.audio_samplerate * self.samplerate)
        factor = target_length / x.size
        x_interp = zoom(x, factor, order=3)
        y_interp = zoom(y, factor, order=3)

        # trim back to self.subsampling points
        new_start_idx = int((start_idx - pad_start_idx) / self.audio_samplerate * self.samplerate)
        x_interp = x_interp[new_start_idx : new_start_idx + self.subsampling]
        y_interp = y_interp[new_start_idx : new_start_idx + self.subsampling]

        # add voltage jitter
        if self.jitter > 0:
            x_noise, y_noise = self._get_jitter_noise(self.jitter_corr)
            x_interp += x_noise * self.jitter
            y_interp += y_noise * self.jitter

        return self._map_to_pixels(x_interp, y_interp)

    def _downscale_to_base_res(self, image):
        """
        (GPU) Downscales a 2D image to oscilloscope width and height.
        """
        
        factor = self.upscaling
        height, width = self.height, self.width

        # downscale with reshape+mean
        downscaled_image = image.reshape(height, factor, width, factor).mean(axis=(1,3))

        return downscaled_image

    def _init_scope(self):
        """
        (GPU) Initialize the analog oscilloscope phosphor state.
        """

        self.frame = -1
        size = (self.height*self.upscaling, self.width*self.upscaling)

        # preallocate physical and camera states
        self.phosphor_state = cp.zeros(size, dtype=cp.float32)
        self.camera_state = cp.zeros(size, dtype=cp.float32)
        self.camera_state_flash = cp.zeros(size, dtype=cp.float32)

    def _draw_beam(self, result_image, x, y, intensity):
        """
        (GPU) Draws electron beam with an arbitrary intensity curve on top of a 2D image.
        Accelerated using CUDA for fast addition.
        """

        threadsperblock = 256
        blockspergrid = math.ceil(x.shape[0] / threadsperblock) # (subsampling steps / no. of threads)
        
        _draw_beam_cuda[blockspergrid, threadsperblock](result_image, x, y, intensity)

    def _advance_frame(self):
        """
        (GPU) Advances the state of the analog oscilloscope by a single frame.
        """

        self.frame += 1

        # apply exponential decay to the previous frame
        # and snap any dark pixels to 0
        self.phosphor_state *= self.decay_factor
        self.phosphor_state[self.phosphor_state < 1e-5] = 0.

        # get x, y coordinates for the new frame
        x, y = self._get_frame_data(self.frame)

        # --- (1) CALCULATE CAMERA STATE ---
        x_trimmed = x[:self.shutter_steps]
        y_trimmed = y[:self.shutter_steps]

        if self.dual_color:
        
            # draw "glow" beam
            cp.copyto(self.camera_state, self.phosphor_state)
            self._draw_beam(self.camera_state, x_trimmed, y_trimmed, self.video_glow_curve)

            # draw "flash" beam
            self.camera_state_flash.fill(0) # reset fluorescence canvas
            self._draw_beam(self.camera_state_flash, x_trimmed, y_trimmed, self.video_flash_curve)

        else:

            # draw combined beam
            cp.copyto(self.camera_state, self.phosphor_state)
            self._draw_beam(self.camera_state, x_trimmed, y_trimmed, self.video_combined_curve)

        # --- (2) CALCULATE PHOSPHOR STATE ---
        self._draw_beam(self.phosphor_state, x, y, self.state_curve)

    def _get_rendered_frame(self, exposure, gamma, bgr_base_color, bgr_flash_color):
        """
        (GPU) Converts the linear video image to gamma-corrected, colorized 8-bit image.
        """

        # get the linear frame and convert it to color
        linear = self._downscale_to_base_res(self._apply_blur(self.camera_state + self.background_level))

        # apply color using broadcasting (height, width) -> (height, width, 3)
        colored = linear[..., cp.newaxis] * bgr_base_color 

        # if dual color logic is on, add the "flash" contribution separately
        if self.dual_color:

            # get the separate camera output frames and multiply them by exposure
            linear_flash = self._downscale_to_base_res(self._apply_blur(self.camera_state_flash))
            colored_flash = linear_flash[..., cp.newaxis] * bgr_flash_color
            colored += colored_flash

        # add monochromatic, multiplicative grain
        if self.grain > 0:
            # gaussian dist around 1.0 with sigma = self.grain
            grain = cp.random.normal(1.0, self.grain, (self.height, self.width), dtype=cp.float32)
            grain = cp.clip(grain, 0., None)
            colored *= grain[..., cp.newaxis]

        colored *= exposure # apply exposure
        clipped = cp.tanh(colored) # soft clip to 0.0, 1.0
        gamma_corrected = clipped ** (1.0 / gamma) # apply gamma-correction
        image = cp.clip(255 * gamma_corrected, 0, 255).astype(cp.uint8) # convert to 8-bit image

        return image.get()

    def render(self, output_path, save_as_single_frames = False, start_at_frame=None, end_at_frame=None, exposure=1.0, gamma=2.2, color="#19ff3f", flash_color=None):

        # --- (1) HANDLE COLORS ---
        # handle colors
        if flash_color is None: flash_color = color
        bgr_base = self._hex_to_bgr(color)
        bgr_flash = self._hex_to_bgr(flash_color)

        # check if colors are identical and enable/disable dual color logic
        if not cp.array_equal(bgr_base, bgr_flash):
            if not self.dual_color:
                print("base_color and flash_color are different. Enabling dual color logic.")
                self.dual_color = True
        else:
            if self.dual_color:
                print("base_color and flash_color are equal. Disabling dual color logic.")
                self.dual_color = False

        # --- SETUP ---

        # reset simulation
        self._init_scope()

        # handle start and end frames
        if not start_at_frame is None:
            self.frame = start_at_frame - 1
        if end_at_frame is None:
            end_at_frame = int(self.total_samples / self.subsampling)

        total_frames = end_at_frame - (self.frame + 1)

        print(f"Starting {self.width}x{self.height} render at {self.fps} FPS, subsampling {self.subsampling}, upscaling {self.upscaling}, glow kernel {self.glow_kernel.size}.\nVideo duration: {self.duration:.1f} s. Total frames to render: {total_frames}")

        if save_as_single_frames:

            # create frames dir if doesn't exist
            os.makedirs(output_path, exist_ok=True)

        else:

            # setup ffmpeg and open pipe
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-loglevel', 'warning',                     # disable logging
                '-s', f'{self.width}x{self.height}',        # resolution
                '-pix_fmt', 'bgr24', '-r', str(self.fps),   # input pixel format and fps
                '-i', '-'                                   # video input
            ]

            # if starting in the middle of the song
            # cut the audio accordingly
            if not start_at_frame is None:
                start_at_time = start_at_frame / self.fps
                ffmpeg_cmd.extend(['-ss', str(start_at_time)])

            ffmpeg_cmd.extend([
                '-i', self.audio_filename,                  # audio input
                '-c:v', 'h264_nvenc', '-preset', 'p6',      # nvenc encoder
                '-cq', '16', '-pix_fmt', 'yuv420p',         # video quality and output pixel format
                '-c:a', 'aac', '-b:a', '320k', '-shortest', # audio quality
                output_path                                 # output path
            ])
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # --- RENDER ---

        start_time = time.time()

        try:

            for f in range(total_frames):
                
                # advance frame and get image
                self._advance_frame()
                image = self._get_rendered_frame(exposure, gamma, bgr_base, bgr_flash)

                if save_as_single_frames:
                    # get path and save with OpenCV
                    import cv2
                    filename = os.path.join(output_path, f"{f:05d}.png")
                    cv2.imwrite(filename, image)
                else:
                    # pipe raw bytes to ffmpeg
                    process.stdin.write(image.tobytes())

                # print progress
                if f % 50 == 0 and f != 0:
                    elapsed = time.time() - start_time
                    eta = elapsed/f * (total_frames - f)
                    speed = f/elapsed / self.fps
                    print(f"Progress: {f}/{total_frames} ({100*f/total_frames:.1f}%) | {speed:.2f}x realtime | elapsed {elapsed:.1f} s | ETA {eta:.1f} s")
        
        finally:
           
            if not save_as_single_frames:
                # close pipe and wait for ffmpeg to finish!!
                process.stdin.close()
                process.wait()

        print(f"Render complete in {time.time()-start_time:.1f} s! Saved in {output_path}.")