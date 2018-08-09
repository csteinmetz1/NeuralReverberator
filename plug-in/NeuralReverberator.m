classdef NeuralReverberator < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('nverb_stereo.mat') % contains interleved stereo outputs
        nverbFs = 16000;                 % 16 kHz sampling rate 
        nverbLength = 32512;             % duration of RIR in samples
        nverbTime = 2.032;               % duration of RIR in seconds
    end     
    
    properties
        inputGain = -3;       % dB factor to scale input signal
        lowpassFc = 20000.0;  % cutoff freq of 2nd order lowpass filter
        lowpass = false;      % control to enable/disable lowpass filter
        highpassFc = 20.0;    % cutoff freq of 2nd order highpass filter
        highpass = false;     % control to enable/disable highpass filter
        predelay = 0.0;       % delay in ms applied to the wet signal
        x = 0;                % first latent dimension 
        y = 0;                % second latent dimension 
        z = 0;                % third latent dimension 
        width = 0;            % control impulse response applied to left and right channels      
        mix = 100;            % percent of wet to dry signal in output
        resampling = true;    % control resampling to match operating sample rate
    end
    
    properties (Constant)
        PluginInterface = audioPluginInterface(...
            'InputChannels',2,...
            'OutputChannels',2,...
            'PluginName','NeuralReverberator',...
            audioPluginParameter('inputGain','DisplayName', 'Input Gain','Label', 'dB','Mapping', {'pow', 1/3, -140, 12}),...
            audioPluginParameter('x','DisplayName','x','Mapping',{'int',0,9}),...
            audioPluginParameter('y','DisplayName','y','Mapping',{'int',0,9}),...
            audioPluginParameter('z','DisplayName','z','Mapping',{'int',0,9}),...
            audioPluginParameter('width','DisplayName','Width','Mapping',{'int',-4,4}),...
            audioPluginParameter('predelay','DisplayName','Pre-Delay','Label','ms','Mapping',{'lin',0,100.0}),...
            audioPluginParameter('lowpass','DisplayName','Lowpass','Mapping', {'enum','Disable','Enable'}),...
            audioPluginParameter('lowpassFc','DisplayName','Lowpass Fc','Label','Hz','Mapping',{'log',500.0,20000.0}),...
            audioPluginParameter('highpass','DisplayName','Highpass','Mapping', {'enum','Disable','Enable'}),...
            audioPluginParameter('highpassFc','DisplayName','Highpass Fc','Label','Hz','Mapping',{'log',20.0,5000.0}),...
            audioPluginParameter('mix','DisplayName','Mix','Label','%','Mapping',{'lin',0,100}),...
            audioPluginParameter('resampling','DisplayName','Resampling','Mapping', {'enum','Disable','Enable'}));
    end
    %----------------------------------------------------------------------
    % private properties
    %----------------------------------------------------------------------
    properties(Access = private)
        % Highpass filter coefficients 
        HPFNum
        HPFDen
        HPFState = zeros(2);
        
        % Lowpass filter coefficients 
        LPFNum
        LPFDen
        LPFState = zeros(2);
                   
        % MATLAB System Objects
        pFracDelay          % delay for pre-delay
        pFIRLeft            % left channel convolutional Filter
        pFIRRight           % right channel convolution Filter
        pFIRRateConv32k     % 16 kHz to 32 kHz SRC
        pFIRRateConv44k     % 44.1 kHz to 32 kHz SRC
        pFIRRateConv48k     % 48 kHz to 32 kHz SRC
        pFIRRateConv96k     % 96 kHz to 32 kHz SRC
        
        % Paramter update flags 
        updateHPF = false;  % update HPF coefficeints
        updateLPF = false;  % update HPF coefficeints
        updateFIR = false;  % update FIR (left and right)
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateFIR
                % get proper RIRs for given parameters
                [left, right] = getRIR(plugin);
                
                % resample to match input (for supported Fs)
                left = resample(plugin, left, getSampleRate(plugin));
                right = resample(plugin, right, getSampleRate(plugin));

                % normalize to -12dB
                left = normalize(plugin, left, -12);
                right = normalize(plugin, right, -12);
                
                % update FIR filter coefficients
                plugin.pFIRLeft.Numerator = left;
                plugin.pFIRRight.Numerator = right;
                setUpdateFIR(plugin,false)
            end
            
            % update HPF coefficients
            if plugin.updateHPF
                [plugin.HPFNum, plugin.HPFDen] = calcHPFCoefs(plugin);
                setUpdateHPF(plugin,false)
            end
            
            % update LPF coefficients
            if plugin.updateLPF              
                [plugin.LPFNum, plugin.LPFDen] = calcLPFCoefs(plugin);
                setUpdateLPF(plugin,false)
            end
            
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            dry = 10.^(plugin.inputGain/20)*u;
            
            % Convolve RIR with left and right channels
            wetLeft = plugin.pFIRLeft(dry(:,1));
            wetRight = plugin.pFIRRight(dry(:,2));

            % Pack left and right channels into 2D array
            wet = [wetLeft, wetRight];
            
            % Apply HPF and LPF filter
            if plugin.highpass
                [wet, plugin.HPFState] = filter(plugin.HPFNum, plugin.HPFDen,...
                                                wet, plugin.HPFState);       
            end
            if plugin.lowpass
                [wet, plugin.LPFState] = filter(plugin.LPFNum, plugin.LPFDen,...
                                                wet, plugin.LPFState);
            end
            
            % add Pre-delay
            delaySamples = (plugin.predelay/1000) * getSampleRate(plugin);
            wet = plugin.pFracDelay(wet, delaySamples);
     
            % mix wet and dry signals together
            y = ((1-(plugin.mix/100)) * dry) + ((plugin.mix/100) * wet);
        end

        function setupImpl(plugin, ~)    
            % initialize supported sample rate converters
            plugin.pFIRRateConv32k = dsp.FIRRateConverter(2,1);
            plugin.pFIRRateConv44k = dsp.FIRRateConverter(11,4);
            plugin.pFIRRateConv48k = dsp.FIRRateConverter(3,1);
            plugin.pFIRRateConv96k = dsp.FIRRateConverter(6,1);
            
            % initialize HPF and LPF filters
            [plugin.HPFNum, plugin.HPFDen] = calcHPFCoefs(plugin);
            [plugin.LPFNum, plugin.LPFDen] = calcLPFCoefs(plugin);
            
            % constant buffer of 195,072 samples
            numerator = zeros(1, plugin.nverbTime * 96000); 

            % create FIR filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', numerator,...
                'PartitionForReducedLatency', true, 'PartitionLength', 2048);
            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', numerator,...
                'PartitionForReducedLatency', true, 'PartitionLength', 2048);
            
            % Create fractional delay
            plugin.pFracDelay = dsp.VariableFractionalDelay('MaximumDelay',192000*.3);
            
            % FIR coefficients will be set on first step
            setUpdateFIR(plugin, true)
        end

        function resetImpl(plugin)
            % rest state of system objects
            reset(plugin.pFracDelay);
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);  
            reset(plugin.pFIRRateConv32k);
            reset(plugin.pFIRRateConv44k);
            reset(plugin.pFIRRateConv48k);
            reset(plugin.pFIRRateConv96k);
            
            % reset intial conditions for filters
            plugin.HPFState = zeros(2);
            plugin.LPFState = zeros(2);
            
            % request update of filters
            setUpdateFIR(plugin,true)
            setUpdateHPF(plugin,true)
            setUpdateLPF(plugin,true)
        end
    end
    %----------------------------------------------------------------------
    % private methods
    %----------------------------------------------------------------------
    methods (Access = private)
        %----------------- Parameter Change Flags -------------------------
        function setUpdateHPF(plugin,flag)
            plugin.updateHPF = flag;
        end
        function setUpdateLPF(plugin,flag)
            plugin.updateLPF = flag;
        end
        function setUpdateFIR(plugin,flag)
            plugin.updateFIR = flag;
        end
        %------------------- Signal Processing Utils-----------------------
        function y = padVector(~, x, outputLength)
            inputLength = length(x);
            padLength = outputLength - inputLength;
            y = [x, zeros(1, padLength)];
        end
        function [left, right] = getRIR(plugin)
            % determine left and right channel RIR index
            leftIndex = (plugin.x * 200 + plugin.y * 20 + plugin.z * 2) + 1;
            rightIndex = leftIndex + plugin.width;
            
            % perform checking on index to ensure its in range
            if rightIndex > 2000
                rightIndex = 2000; % max valid index
            elseif rightIndex < 1
               rightIndex = 1;     % min valid index
            end

            % extract proper RIRs as column slices - return as row vector
            left = transpose(plugin.nverb.RIRAudio(:,leftIndex));
            right = transpose(plugin.nverb.RIRAudio(:,rightIndex));
        end
        function [y] = resample(plugin, x, outFs)
            if plugin.resampling
                if   outFs == 32000
                    y = plugin.pFIRRateConv32k(transpose(x));
                    y = transpose(y);
                    y = padVector(plugin, y, plugin.nverbTime * 96000);
                elseif outFs == 44100
                    y = plugin.pFIRRateConv44k(transpose(x));
                    y = transpose(y);
                    y = padVector(plugin, y, plugin.nverbTime * 96000);
                elseif outFs == 48000 
                    y = plugin.pFIRRateConv48k(transpose(x));
                    y = transpose(y);
                    y = padVector(plugin, y, plugin.nverbTime * 96000);
                elseif outFs == 96000
                    y = plugin.pFIRRateConv96k(transpose(x));
                    y = transpose(y);
                    y = padVector(plugin, y, plugin.nverbTime * 96000);
                else
                    y = padVector(plugin, x, plugin.nverbTime * 96000);
                end
            else
                y = padVector(plugin, x, plugin.nverbTime * 96000);
            end
        end
        function [y] = normalize(~, x, peak)
            currentPeak = max(x);
            gain = 10^(peak/20) / currentPeak;
            y = gain * x;            
        end
        %--------------------- Coefficient Cooking ------------------------
        function [b, a] = calcLPFCoefs(plugin)
            % initial values
            w0 = 2 * pi * (plugin.lowpassFc/getSampleRate(plugin));
            Q = 1/sqrt(2);
            alpha = (sin(w0) / (2 * Q));
            % coefs calculation
            b0 = (1 - cos(w0))/2;
            b1 = (1 - cos(w0));
            b2 = (1 - cos(w0))/2;
            a0 =  1 + alpha;
            a1 = -2 * cos(w0);
            a2 =  1 - alpha;
            % normalized output coefs
            b = [b0/a0, b1/a0, b2/a0];
            a = [a0/a0, a1/a0, a2/a0];
        end
        function [b, a] = calcHPFCoefs(plugin)
            % initial values
            w0 = 2 * pi * (plugin.highpassFc/getSampleRate(plugin));
            Q = 1/sqrt(2);
            alpha = (sin(w0) / (2 * Q));
            % coef calculation
            b0 =  (1 + cos(w0))/2;
            b1 = -(1 + cos(w0));
            b2 =  (1 + cos(w0))/2;
            a0 =   1 + alpha;
            a1 =  -2 * cos(w0);
            a2 =   1 - alpha;
            % normalized output coefs
            b = [b0/a0, b1/a0, b2/a0];
            a = [a0/a0, a1/a0, a2/a0];
        end
    end
    %----------------------------------------------------------------------
    % setter and getter methods
    %----------------------------------------------------------------------
    methods 
        function set.lowpassFc(plugin, val)
            plugin.lowpassFc = val;
            setUpdateLPF(plugin, true);
        end
        function val = get.lowpassFc(plugin)
            val = plugin.lowpassFc;
        end
        function set.highpassFc(plugin, val)
            plugin.highpassFc = val;
            setUpdateHPF(plugin, true);
        end
        function val = get.highpassFc(plugin)
            val = plugin.highpassFc;
        end
        function set.x(plugin, val)
            plugin.x = val;
            setUpdateFIR(plugin, true);
        end
        function val = get.x(plugin)
            val = plugin.x;
        end  
        function set.y(plugin, val)
            plugin.y = val;
            setUpdateFIR(plugin, true);
        end
        function val = get.y(plugin)
            val = plugin.y;
        end      
        function set.z(plugin, val)
            plugin.z = val;
            setUpdateFIR(plugin, true);
        end
        function val = get.z(plugin)
            val = plugin.z;
        end
        function set.width(plugin, val)
            plugin.width = val;
            setUpdateFIR(plugin, true);
        end
        function val = get.width(plugin)
            val = plugin.width;
        end
        function set.resampling(plugin, val)
            plugin.resampling = val;
            setUpdateFIR(plugin, true);
        end
        function val = get.resampling(plugin)
            val = plugin.resampling;
        end
    end  
end