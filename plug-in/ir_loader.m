cd /Volumes/HDMETZ1/Datasets/nverb/pre_compute/

%%
ir_files = dir('*.wav');
ir_audio = zeros(65280,length(ir_files));

for i=1:length(ir_files)
    [y, Fs] = audioread(ir_files(i).name);
    ir_audio(:,i) = transpose(y);
end

save ('nverb.mat', 'ir_audio') 
