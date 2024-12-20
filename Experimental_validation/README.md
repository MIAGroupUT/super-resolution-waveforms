# Experimental super-resolution

This folder contains files used for processing and displaying experimental data. See Section II-F of the article.

The measurement protocol is described in the data respository (at `..\SRML-1D-pulse-types\Results\Experiments`).

## File overview
### Main RF simulator files
* `get_noise_params.m` Find the characteristics of experimental noise, such as RMS noise and signal-to-noise ratio.
* `make_video.m` Create a video of the acquired and processed frames.
* `process_exp_data.m` Process the Verasonics outputs to .mat files which can be used for super-resolution.
* `show_exp_results.m` Create an image displaying a diffraction-limited and super-resolved image.

### Folders:
* 📂 `extra_files`	Information of the bubble sizes used in the experiments, impulse responses and measuring system.

