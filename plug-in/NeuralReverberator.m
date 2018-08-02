classdef NeuralReverberator < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('nverb_stereo.mat')
        nverbFs = 16000;
        nverbLength = 65280;
        PartitionSize = 1024;
    end     
    
    properties
        InputGain = -6;
        Lowpass = 20000.0;
        Highpass = 20.0;
        PreDelay = 0.0;
        a = 1;
        b = 1;
        c = 1;
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
            audioPluginParameter('PreDelay','DisplayName','Pre-Delay','Label','ms','Mapping',{'lin',-50.0,300.0}),...
            audioPluginParameter('a','DisplayName','A','Mapping',{'int',1,10}),...
            audioPluginParameter('b','DisplayName','B','Mapping',{'int',1,10}),...
            audioPluginParameter('c','DisplayName','C','Mapping',{'int',1,10}),...
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
        
        p
        q
       
        UpdateHPF = false;
        UpdateLPF = false;
        UpdateRIRAudio = false;
        ResampleAudio = false;
        
        pFIRRateConv
        pFIRLeft
        pFIRRight
        pFracDelay
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            
            if plugin.UpdateRIRAudio
                % Get proper RIRs for given parameters
                [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);

                % Upsample to match input - not sure how to handle this yet
                if plugin.ResampleAudio
                    FIRRateConv = dsp.FIRRateConverter('InterpolationFactor', plugin.p,...
                                                       'DecimationFactor', plugin.q);
                    resampledRIRAudioLeft = FIRRateConv(RIRAudioLeft);
                    resampledRIRAudioRight = FIRRateConv(RIRAudioRight);
                end

                % Update FIR filter
                plugin.pFIRLeft.Numerator = resampledRIRAudioLeft;
                plugin.pFIRRight.Numerator = resampledRIRAudioRight;
                setUpdateRIRAudio(plugin,false)
            end
            
            if plugin.UpdateHPF
                % Update HPF coefficients
                [plugin.HPFNum, plugin.HPFDen] = calculateHPFCoefficients(plugin);
                setUpdateHPF(plugin,false)
            end
            
            if plugin.UpdateLPF              
                % Update LPF coefficients
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
            
            % Set sample rate converter factor
            [plugin.p, plugin.q] = calculateResampleFactor(plugin);

            % Initialize HPF and LPF filters
            [plugin.HPFNum, plugin.HPFDen] = calculateHPFCoefficients(plugin);
            [plugin.LPFNum, plugin.LPFDen] = calculateLPFCoefficients(plugin);
            
            % Create frequency domain filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudioLeft,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);

            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudioRight,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFracDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
        end

        function resetImpl(plugin)
            
            % Release since numerator size changes on sample rate change
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);
            
            % Initialize reverb
            %[RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);
                        
            % Upsample to match input - not sure how to handle this yet
            %resampledRIRAudioLeft = resampleRIR(plugin, RIRAudioLeft);
            %resampledRIRAudioRight = resampleRIR(plugin, RIRAudioRight);
            
            % Re-init frequency domain filters for convolution 
            %plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudioLeft,...
            %    'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);

            %plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', RIRAudioRight,...
            %    'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            reset(plugin.pFracDelay);
            plugin.HPFState = zeros(2);
            plugin.LPFState = zeros(2);
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
        function [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin)
            RIRIndex = (plugin.a * 200 + plugin.b * 20 + plugin.c * 2);
            RIRAudioLeft = transpose(plugin.nverb.RIRAudio(:,RIRIndex));
            RIRAudioRight = transpose(plugin.nverb.RIRAudio(:,RIRIndex+1));
        end
        function [p, q] = calculateResampleFactor(plugin)
            if     getSampleRate(plugin) == 32000
                p = 2;
                q = 1;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 44100
                p = 441;
                q = 160;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 48000
                p = 3;
                q = 1;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 96000
                p = 6;
                q = 1;
                plugin.ResampleAudio = true;
            elseif getSampleRate(plugin) == 192000
                p = 12;
                q = 1;
                plugin.ResampleAudio = true;
            else
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
        function set.a(plugin, val)
            plugin.a = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.a(plugin)
            val = plugin.a;
        end  
        function set.b(plugin, val)
            plugin.b = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.b(plugin)
            val = plugin.b;
        end      
        function set.c(plugin, val)
            plugin.c = val;
            setUpdateRIRAudio(plugin, true);
        end
        function val = get.c(plugin)
            val = plugin.c;
        end
    end  
end