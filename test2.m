clear
close all
% [sig,fs]=audioread('recordings/rec_20261018_194104.wav'); %control1
% [sig,fs]=audioread('recordings/rec_20261018_194149.wav'); %control2
% [sig,fs]=audioread('recordings/rec_20261018_194231.wav'); %control3
% [sig,fs]=audioread('recordings/rec_20261018_193657.wav'); %ear1
% [sig,fs]=audioread('recordings/rec_20261018_194311.wav'); %ear2
% [sig,fs]=audioread('recordings/rec_20261018_194335.wav'); %ear3
% [sig,fs]=audioread('recordings/rec_20261018_195022.wav'); %ear4
% [sig,fs]=audioread('recordings/rec_20261018_195215.wav'); %ear5

segLength=18;
[env,~]=envelope(sig,3000,'peak');
thr = 0.1;
onset = find(conv(double(env > thr), ones(1000,1), 'valid') == 1000, 1)+fs;
endset=onset+(fs*segLength)-1;
sig=sig(onset:endset);

segs=reshape(sig, fs, segLength);
% f1=1557;
% f2=1900;
f1=2377;
f2=2900;
oae=(2*f1-f2)+1;
tol=2; w=200;

segPow=abs(fft(segs)).^2;  % per-segment power spectra, phase discarded

nAvg=2:segLength;
snr=zeros(size(nAvg));
for k=1:numel(nAvg)
    spec=sqrt(mean(segPow(:,1:nAvg(k)),2));
    noise=mean(spec([oae-w-tol:oae-tol, oae+tol:oae+tol+w]));
    snr(k)=10*log10(spec(oae))-10*log10(noise);
end
table(nAvg', snr', 'VariableNames', {'seconds','snr_dB'})

figure
subplot(121)
plot(nAvg, snr, '-o');
xlabel('Seconds averaged'); ylabel('SNR (dB)');
ylim([-10,10])
yline(0)
xlim([0,20])

spec2=mag2db(spec);
subplot(122)
hold on
plot(spec2);
scatter(oae,spec2(oae),'filled')
xlim([0,4000])

