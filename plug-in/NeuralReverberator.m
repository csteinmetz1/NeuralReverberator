classdef NeuralReverberator < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('nverb_stereo.mat')
        nverbFs = 16000;
        nverbLength = 32512;
        nverbTime = 2.032;
        PartitionSize = 2048;
    end     
    
    properties
        InputGain = -3;
        Lowpass = 20000.0;
        Highpass = 20.0;
        PreDelay = 0.0;
        A = 0;
        B = 0;
        C = 0;
        Width = 0;
        Mix = 100;
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
            audioPluginParameter('Mix','DisplayName','Mix','Label','%','Mapping',{'lin',0,100}));
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
        
        pFracDelay 
        
        pFIRLeft
        pFIRRight
        
        pFIRRateConv32k
        pFIRRateConv44k
        pFIRRateConv48k
        pFIRRateConv96k

    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            
            if plugin.UpdateRIRAudio
                % Get proper RIRs for given parameters
                [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);
                
                % Upsample to match inpu
                resampledRIRAudioLeft = resample(plugin, RIRAudioLeft, getSampleRate(plugin));
                resampledRIRAudioRight = resample(plugin, RIRAudioRight, getSampleRate(plugin));

                % Normalize audio to -12dB
                normalizedRIRAudioLeft = normalize(plugin, resampledRIRAudioLeft, -12);
                normalizedRIRAudioRight = normalize(plugin, resampledRIRAudioRight, -12);
                
                % Update FIR filter of the current sample rate
                plugin.pFIRLeft.Numerator = normalizedRIRAudioLeft;
                plugin.pFIRRight.Numerator = normalizedRIRAudioRight;
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
            y = ((1-(plugin.Mix/100)) * u) + ((plugin.Mix/100) * wet);
        end

        function setupImpl(plugin, u)
                       
            % Initialize supported sample rate converters
            plugin.pFIRRateConv32k = dsp.FIRRateConverter(2,1);
            plugin.pFIRRateConv44k = dsp.FIRRateConverter(3,1); % set to 48k for now
            plugin.pFIRRateConv48k = dsp.FIRRateConverter(3,1);
            plugin.pFIRRateConv96k = dsp.FIRRateConverter(6,1);
            
            % Initialize HPF and LPF filters
            [plugin.HPFNum, plugin.HPFDen] = calculateHPFCoefficients(plugin);
            [plugin.LPFNum, plugin.LPFDen] = calculateLPFCoefficients(plugin);
            
            RIRAudio = zeros(1, plugin.nverbTime * 96000); % constant buffer of 195,072 samples

            % Create frequency domain filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudio,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudio,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFracDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
            
            setUpdateRIRAudio(plugin, true)
        end

        function resetImpl(plugin)
            
            % Rest state of system objects
            reset(plugin.pFracDelay);
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);  
            reset(plugin.pFIRRateConv32k);
            reset(plugin.pFIRRateConv44k);
            reset(plugin.pFIRRateConv48k);
            reset(plugin.pFIRRateConv96k);
            
            % Reset intial conditions for filters
            plugin.HPFState = zeros(2);
            plugin.LPFState = zeros(2);
            
            setUpdateRIRAudio(plugin,true)
            setUpdateHPF(plugin,true)
            setUpdateLPF(plugin,true)
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
        
        function paddedArray = pad1DArray(~, array, outputArrayLength)
            inputArrayLength = length(array);
            padLength = outputArrayLength - inputArrayLength;
            paddedArray = [array, zeros(1, padLength)];
        end
        
        function [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin)
            
            % determine left and right channel RIR indexes
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
        
        function [resampledRIRAudio] = resample(plugin, RIRAudio, outputFs)
            %if outputFs == 48000
            %    resampledRIRAudio = plugin.pFIRRateConv48k(transpose(RIRAudio));
            %    resampledRIRAudio = transpose(resampledRIRAudio);
            %else
            %    resampledRIRAudio = RIRAudio;
            %end
            resampledRIRAudio = pad1DArray(plugin, RIRAudio, plugin.nverbTime * 96000);
        end
        
        function [normalizedRIRAudio] = normalize(~, RIRAudio, peak)
            currentPeak = max(RIRAudio);
            gain = 10^(peak/20) / currentPeak;
            normalizedRIRAudio = gain * RIRAudio;            
        end
        
        function [b, a] = calculateLPFCoefficients(plugin)
            w0 = 2 * pi * (plugin.Lowpass/getSampleRate(plugin));
            Q = 1/sqrt(2);
            alpha = (sin(w0) / (2 * Q));
            
            b0 = (1 - cos(w0))/2;
            b1 = (1 - cos(w0));
            b2 = (1 - cos(w0))/2;
            a0 =  1 + alpha;
            a1 = -2 * cos(w0);
            a2 =  1 - alpha;
            
            b = [b0/a0, b1/a0, b2/a0];
            a = [a0/a0, a1/a0, a2/a0];
        end
        function [b, a] = calculateHPFCoefficients(plugin)
            w0 = 2 * pi * (plugin.Highpass/getSampleRate(plugin));
            Q = 1/sqrt(2);
            alpha = (sin(w0) / (2 * Q));
           
            b0 =  (1 + cos(w0))/2;
            b1 = -(1 + cos(w0));
            b2 =  (1 + cos(w0))/2;
            a0 =   1 + alpha;
            a1 =  -2 * cos(w0);
            a2 =   1 - alpha;
            
            b = [b0/a0, b1/a0, b2/a0];
            a = [a0/a0, a1/a0, a2/a0];
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