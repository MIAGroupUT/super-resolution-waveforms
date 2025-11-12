function V = receiveTransferFunction(p,H,Fs)
% Convert received pressure to a voltage signal.

% Check sample rate
if Fs ~= 250e6
    error('This function only works for a sampling frequency of 250 MHz')
end

% Resample transfer function
N = length(p);
M = length(H);
H_r = resample(H,N,M);

% Apply transfer function
pfft = fft(p);
Vfft = pfft.*H_r;
V = real(ifft(Vfft));

end