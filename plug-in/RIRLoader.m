% point to location of generated Room Impulse Responses (RIRs)
%cd ../train/pre_compute/
cd /Volumes/HDMETZ1/Datasets/nverb/train/pre_compute3

% Find all generated (RIRs)
RIRFiles = dir('*.wav');

% Load first RIR to get sample rate and length
[y, Fs] = audioread(RIRFiles(1).name);
RIRSampleRate = Fs;
RIRLength = length(y);

% Generate matrix where each column represents a RIR
RIRAudio = zeros(RIRLength,length(RIRFiles));

% Load all RIRs storing into matrix
for i=1:length(RIRFiles)
    [y, Fs] = audioread(RIRFiles(i).name);
    RIRAudio(:,i) = transpose(y);
    fprintf('Collecting RIR %s\n',RIRFiles(i).name(1:4))
end

% Save matrix to file to be loaded by plug-in
fprintf('Saving RIR files to nverb_stereo.mat\n')
save ('nverb_stereo.mat', 'RIRAudio') 
