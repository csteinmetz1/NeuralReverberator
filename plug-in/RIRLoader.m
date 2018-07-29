% point to location of generated Room Impulse Responses (RIRs)
cd ../train/pre_compute/

% Find all generated (RIRs)
RIRFiles = dir('*.wav');

% Load first RIR to get sample rate and length
[y, Fs] = audioread(RIRFiles(1).name);
RIRSampleRate = Fs;
RIRLength = length(y);

% Generate matrix where each column represents a RIR
RIRAudio = zeros(65280,length(ir_files));

% Load all RIRs storing into matrix
for i=1:length(RIRFiles)
    [y, Fs] = audioread(RIRFiles(i).name);
    RIRAudio(:,i) = transpose(y);
end

% Save matrix to file to be loaded by plug-in
save ('nverb.mat', 'RIRAudio') 
