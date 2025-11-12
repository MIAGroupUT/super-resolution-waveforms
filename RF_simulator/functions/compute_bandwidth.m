function Df = compute_bandwidth(Tfit,Fs)
% Compute bandwidth of transducer with transfer function Tfit.

% Transfer function on dB scale:
Y_dB = 20*log10(abs(Tfit)/max(abs(Tfit)));
N = length(Y_dB);

% Find the -6 dB bandwidth of the transducer:
I1 = find(Y_dB(1:round(N/2))>-6,1);         % Left  -6 dB point
I2 = find(Y_dB(1:round(N/2))>-6,1,'last');  % Right -6 dB point
Df = (I2 - I1)/N*Fs;                        % Bandwidth (-6 dB) (Hz)

Dlow = I1/(N*1e6)*Fs;
Dhigh = I2/(N*1e6)*Fs;

% Display transducer bandwidth
disp("Transducer (receiver) -6dB bandwidth: "+Dlow+"-"+Dhigh+" MHz")

end