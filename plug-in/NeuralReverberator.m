classdef NeuralReverberator < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('nverb_stereo.mat')
        nverbFs = 16000;
        nverbLength = 65280;
        PartitionSize = 2048;
    end     
    
    properties
        InputGain = -6;
        Lowpass = 20000.0;
        Highpass = 20.0;
        PreDelay = 0.0;
        A = 1;
        B = 1;
        C = 1;
        Width = 0;
        Mix = 1.0;
    end
    
    properties (Constant)
        PluginInterface = audioPluginInterface(...
            'InputChannels',2,...
            'OutputChannels',2,...
            'PluginName','NeuralReverberator',...
            audioPluginParameter('InputGain','DisplayName', 'Input Gain','Label', 'dB','Mapping', {'pow', 1/3, -140, 12}),...
            audioPluginParameter('Lowpass','DisplayName','Lowpass','Label','Hz','Mapping',{'log',500.0,20000.0}),...
            audioPluginParameter('Highpass','DisplayName','Highpass','Label','Hz','Mapping',{'log',20.0,5000.0}),...
            audioPluginParameter('PreDelay','DisplayName','Pre-Delay','Label','ms','Mapping',{'lin',0,300.0}),...
            audioPluginParameter('A','DisplayName','A','Mapping',{'int',0,9}),...
            audioPluginParameter('B','DisplayName','B','Mapping',{'int',0,9}),...
            audioPluginParameter('C','DisplayName','C','Mapping',{'int',0,9}),...
            audioPluginParameter('Width','DisplayName','Width','Mapping',{'int',-4,4}),...
            audioPluginParameter('Mix','DisplayName','Mix','Label','%','Mapping',{'lin',0,1.0}));
    end
    %----------------------------------------------------------------------
    % Private properties
    %----------------------------------------------------------------------
    properties(Access = private)
        HPFNum
        HPFDen
        HPFState = zeros(2);
        
        LPFNum
        LPFDen
        LPFState = zeros(2);
             
        UpdateHPF = false;
        UpdateLPF = false;
        UpdateRIRAudio = false;
        ResampleAudio = false;
        
        pFIRLeft
        pFIRRight
        pFracDelay
        pFIRRateConv32k
        pFIRRateConv48k
        pFIRRateConv96k
        pFIRRateConv192k
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            
            if plugin.UpdateRIRAudio
                % Get proper RIRs for given parameters
                [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);

                % Upsample (and pad) to match input sample rate
                if plugin.ResampleAudio
                    paddedRIRAudioLeft = resampleRIRAudio(plugin, RIRAudioLeft);
                    paddedRIRAudioRight = resampleRIRAudio(plugin, RIRAudioRight);
                else
                    paddedRIRAudioLeft = pad1DArray(plugin, RIRAudioLeft, 384000);
                    paddedRIRAudioRight = pad1DArray(plugin, RIRAudioRight, 384000);
                end

                % Normalize audio to -12dB
                paddedRIRAudioLeft = normalizeRIRAudio(plugin, paddedRIRAudioLeft, -12);
                paddedRIRAudioRight = normalizeRIRAudio(plugin, paddedRIRAudioRight, -12);
                
                % Update FIR filter
                plugin.pFIRLeft.Numerator = paddedRIRAudioLeft;
                plugin.pFIRRight.Numerator = paddedRIRAudioRight;
                setUpdateRIRAudio(plugin,false)
            end
            
            % Update HPF coefficients
            if plugin.UpdateHPF
                [plugin.HPFNum, plugin.HPFDen] = calculateHPFCoefficients(plugin);
                setUpdateHPF(plugin,false)
            end
            
            % Update LPF coefficients
            if plugin.UpdateLPF              
                [plugin.LPFNum, plugin.LPFDen] = calculateLPFCoefficients(plugin);
                setUpdateLPF(plugin,false)
            end
            
            % Apply input gain
            u = 10.^(plugin.InputGain/20)*u;
            
            % Convolve RIR with left and right channels
            wetLeft = step(plugin.pFIRLeft, u(:,1));
            wetRight = step(plugin.pFIRRight, u(:,2));
            
            % Pack left and right channels into 2D array
            wet = [wetLeft, wetRight];
            
            % Apply HPF and LPF filter
            [wet, plugin.HPFState] = filter(plugin.HPFNum, plugin.HPFDen, wet, plugin.HPFState);           
            [wet, plugin.LPFState] = filter(plugin.LPFNum, plugin.LPFDen, wet, plugin.LPFState);
            
            % Perform Pre-delay
            delaySamples = (plugin.PreDelay/1000) * getSampleRate(plugin);
            wet = plugin.pFracDelay(wet, delaySamples);
     
            % Mix the dry and wet signals together
            y = ((1-plugin.Mix) * u) + (plugin.Mix * wet);
        end

        function setupImpl(plugin, u)
            
            % Initialize reverb
            [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);
            
            % Initialize supported sample rate converters
            plugin.pFIRRateConv32k = dsp.FIRRateConverter('InterpolationFactor', 2, 'DecimationFactor', 1);
            plugin.pFIRRateConv48k = dsp.FIRRateConverter('InterpolationFactor', 3, 'DecimationFactor', 1);
            plugin.pFIRRateConv96k = dsp.FIRRateConverter('InterpolationFactor', 6, 'DecimationFactor', 1);
            plugin.pFIRRateConv192k = dsp.FIRRateConverter('InterpolationFactor', 12, 'DecimationFactor', 1);
            
            % Upsample to match input
            if plugin.ResampleAudio
                paddedRIRAudioLeft = resampleRIRAudio(plugin, RIRAudioLeft);
                paddedRIRAudioRight = resampleRIRAudio(plugin, RIRAudioRight);
            else
                paddedRIRAudioLeft = pad1DArray(plugin, RIRAudioLeft, 384000);
                paddedRIRAudioRight = pad1DArray(plugin, RIRAudioRight, 384000);
            end
            
            % Normalize audio to -12dB
            paddedRIRAudioLeft = normalizeRIRAudio(plugin, paddedRIRAudioLeft, -12);
            paddedRIRAudioRight = normalizeRIRAudio(plugin, paddedRIRAudioRight, -12);
            
            % Initialize HPF and LPF filters
            [plugin.HPFNum, plugin.HPFDen] = calculateHPFCoefficients(plugin);
            [plugin.LPFNum, plugin.LPFDen] = calculateLPFCoefficients(plugin);
            
            % Create frequency domain filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', paddedRIRAudioLeft,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);

            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', paddedRIRAudioRight,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFracDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
        end

        function resetImpl(plugin)
            
            % Rest state of system objects
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);            
            reset(plugin.pFracDelay);
            reset(plugin.pFIRRateConv32k);
            reset(plugin.pFIRRateConv48k);
            reset(plugin.pFIRRateConv96k);
            reset(plugin.pFIRRateConv192k);
            
            % Reset intial conditions for filters
            plugin.HPFState = zeros(2);
            plugin.LPFState = zeros(2);
            
            % Resample RIR Audio on sample rate change
            [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);
                            
            % Upsample to match input
            if plugin.ResampleAudio         
                paddedRIRAudioLeft = resampleRIRAudio(plugin, RIRAudioLeft);
                paddedRIRAudioRight = resampleRIRAudio(plugin, RIRAudioRight);
            else
                paddedRIRAudioLeft = pad1DArray(plugin, RIRAudioLeft, 384000);
                paddedRIRAudioRight = pad1DArray(plugin, RIRAudioRight, 384000);
            end
            
            % Normalize audio to -12dB
            paddedRIRAudioLeft = normalizeRIRAudio(plugin, paddedRIRAudioLeft, -12);
            paddedRIRAudioRight = normalizeRIRAudio(plugin, paddedRIRAudioRight, -12);
            
            % Update FIR filter
            plugin.pFIRLeft.Numerator = paddedRIRAudioLeft;
            plugin.pFIRRight.Numerator = paddedRIRAudioRight;
            setUpdateRIRAudio(plugin,false)
        end
    end
    
    methods (Access = private)
        function setUpdateHPF(plugin,flag)
            plugin.UpdateHPF = flag;
        end
        
        function setUpdateLPF(plugin,flag)
            plugin.UpdateLPF = flag;
        end
        
        function setUpdateRIRAudio(plugin,flag)
            plugin.UpdateRIRAudio = flag;
        end
        
        function paddedArray = pad1DArray(plugin, array, outputArrayLength)
            inputArrayLength = length(array);
            padLength = outputArrayLength - inputArrayLength;
            paddedArray = [array, zeros(1, padLength)];
        end
        
        function [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin)
            % revist this - not currently correct
            RIRLeftIndex = (plugin.A * 200 + plugin.B * 20 + plugin.C * 2) + 1;
            RIRRightIndex = RIRLeftIndex + plugin.Width;
            
            % Perform checking on index to ensure its in range
            if RIRRightIndex > 2000
                RIRRightIndex = 2000;
            elseif RIRRightIndex < 1
                RIRRightIndex = 1;
            end
            
            % Extract proper RIRs as column slices
            RIRAudioLeft = transpose(plugin.nverb.RIRAudio(:,RIRLeftIndex));
            RIRAudioRight = transpose(plugin.nverb.RIRAudio(:,RIRRightIndex));
        end
        
        function [resampledPaddedRIRAudio] = resampleRIRAudio(plugin, RIRAudio)
            if     getSampleRate(plugin) == 32000
                resampledRIRAudio = plugin.pFIRRateConv32k(RIRAudio);
            elseif getSampleRate(plugin) == 44100
                resampledRIRAudio = RIRAudio; % not supported
            elseif getSampleRate(plugin) == 48000
                resampledRIRAudio = plugin.pFIRRateConv48k(RIRAudio);
            elseif getSampleRate(plugin) == 96000
                resampledRIRAudio = plugin.pFIRRateConv96k(RIRAudio);
            elseif getSampleRate(plugin) == 192000
                resampledRIRAudio = plugin.pFIRRateConv192k(RIRAudio);
            else
                resampledRIRAudio = RIRAudio;
            end
            resampledPaddedRIRAudio = pad1DArray(plugin, resampledRIRAudio(end,:), 384000); % not sure how to index this?
        end
        
        function [normalizedRIRAudio] = normalizeRIRAudio(plugin, RIRAudio, peak)
            currentPeak = max(RIRAudio);
            gain = 10^(peak/20) / currentPeak;
            normalizedRIRAudio = gain * RIRAudio;            
        end
        
        function convIndex = calculateResampleFactor(plugin)
            if     getSampleRate(plugin) == 32000
                convIndex = 1;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 44100
                convIndex = 0;
                plugin.ResampleAudio = false; % this factor doesn't work
            elseif getSampleRate(plugin) == 48000
                convIndex = 2;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 96000
                convIndex = 3;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 192000
                convIndex = 4;
                plugin.ResampleAudio = true;
            else
                convIndex = 0;
                plugin.ResampleAudio = false;
            end
        end
        function [b, a] = calculateLPFCoefficients(plugin)
            w0 = 2 * pi * (plugin.Lowpass/getSampleRate(plugin));
            alpha = sin(w0) / (sqrt(2));
            
            b0 = (1 - cos(w0))/2;
            b1 = (1 - cos(w0));
            b2 = (1 - cos(w0))/2;
            a0 =  1 + alpha;
            a1 = -2 * cos(w0);
            a2 =  1 - alpha;
            
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = calculateHPFCoefficients(plugin)
            w0 = 2 * pi * (plugin.Highpass/getSampleRate(plugin));
            alpha = sin(w0) / (sqrt(2)/2);
           
            b0 =  (1 + cos(w0))/2;
            b1 = -(1 + cos(w0));
            b2 =  (1 + cos(w0))/2;
            a0 =   1 + alpha;
            a1 =  -2 * cos(w0);
            a2 =   1 - alpha;
            
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
    end
    
    methods 
        function set.Lowpass(plugin, val)
            plugin.Lowpass = val;
            setUpdateLPF(plugin, true);
        end
        function val = get.Lowpass(plugin)
            val = plugin.Lowpass;
        end
        function set.Highpass(plugin, val)
            plugin.Highpass = val;
            setUpdateHPF(plugin, true);
        end
        function val = get.Highpass(plugin)
            val = plugin.Highpass;
        end
        function set.A(plugin, val)
            plugin.A = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.A(plugin)
            val = plugin.A;
        end  
        function set.B(plugin, val)
            plugin.B = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.B(plugin)
            val = plugin.B;
        end      
        function set.C(plugin, val)
            plugin.C = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.C(plugin)
            val = plugin.C;
        end
        function set.Width(plugin, val)
            plugin.Width = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.Width(plugin)
            val = plugin.Width;
        end
    end  
end