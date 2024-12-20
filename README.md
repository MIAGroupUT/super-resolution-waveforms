# Effect of pulse type on deep-learning-based super-resolution

This repository contains the code for the study of: R. Zorgdrager, N. Blanken, J. M. Wolterink, M. Versluis and G. Lajoinie, "Waveform-Specific Performance of Deep Learning-Based Super-Resolution for Ultrasound Contrast Imaging", in prepraration for the spotlight issue *Breaking the Resolution Barrier in Ultrasound* of IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control.

The code here is based on a clone of https://github.com/MIAGroupUT/SRML-1D.

![Method overview!](Methods.png "Super-resolution localization pipeline")

**Fig. 1** **Methods overview.** **1** *A random distribution of microbubbles is stimulated with the selected pulse using a virtual P4-1 transducer. The simulator computes the local shapes of the pressure wave by accounting for nonlinear propagation in the medium and solves the RP-equation. The received signal by the transducer is used as RF lines for training, validation, and testing.*

The code is organized into four folders:
* 📂 **RF_simulator:** Pulse definition, RF signal simulation, ground truth generation, and optional RF decoding. Section IIA, IIB and, IIC-1 in the article.
* 📂 **Network_pulse_types:** Neural network training and evaluation. Sections IIC-2, IID, IIE-1 in the article.
* 📂 **DelayAndSum:**  Delay-and-sum image reconstruction with unprocessed and deconvolved RF signals. Section IIE-2 in the article.
* 📂 **experimental_validation:** Processing of experimental data. Section II-F in the article.

## Required software

* RF_simulator: MATLAB with Signal Processing Toolbox.
* Network: Python with PyTorch, NumPy, Matplotlib, SciPy
* DelayAndSum: MATLAB with export_fig module.

## Required hardware

* GPU with CUDA cores for network training
* Verasonics Vantage 256 for experimental validation

The code can be found by scanning:

![Super-resolution code!](QR-code-Github.png)
