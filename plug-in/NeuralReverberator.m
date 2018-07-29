classdef NeuralReverberator < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('nverb.mat')
        RIRFs = 16000;
        RIRLength = 65280;
        PartitionSize = 1024;
    end     
    
    properties
        InputGain = -6;
        HFCut = 20000.0;
        LFCut = 1.0;
        PreDelay = 0.0;
        a = 0;
        b = 0;
        c = 0;
        Mix = 1.0;
    end
    
    properties (Constant)
        PluginInterface = audioPluginInterface(...
            'InputChannels',2,...
            'OutputChannels',2,...
            'PluginName','NeuralReverberator',...
            audioPluginParameter('InputGain','DisplayName', 'Input Gain','Label', 'dB','Mapping', {'pow', 1/3, -140, 12}),...
            audioPluginParameter('HFCut','DisplayName','HF Cut','Label','Hz','Mapping',{'log',500.0,20000.0}),...
            audioPluginParameter('LFCut','DisplayName','LF Cut','Label','Hz','Mapping',{'log',1.0,5000.0}),...
            audioPluginParameter('PreDelay','DisplayName','Pre-Delay','Label','ms','Mapping',{'lin',0.0,300.0}),...
            audioPluginParameter('a','DisplayName','A','Mapping',{'int',0,9}),...
            audioPluginParameter('b','DisplayName','B','Mapping',{'int',0,9}),...
            audioPluginParameter('c','DisplayName','C','Mapping',{'int',0,9}),...
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

                % Upsample to match input
                upsampledRIRAudioLeft = upsampleRIR(plugin, RIRAudioLeft);
                upsampledRIRAudioRight = upsampleRIR(plugin, RIRAudioRight);

                % Update FIR filter
                plugin.pFIRLeft.Numerator = upsampledRIRAudioLeft;
                plugin.pFIRRight.Numerator = upsampledRIRAudioRight;
                setUpdateRIRAudio(plugin,false)
            end
            
            if plugin.UpdateHPF
                % Update HPF coefficients
                [plugin.HPFNum, plugin.HPFDen] = butter(2, plugin.LFCut/(getSampleRate(plugin)/2), 'high');
                setUpdateHPF(plugin,false)
            end
            
            if plugin.UpdateLPF
                % Update LPF coefficients
                [plugin.LPFNum, plugin.LPFDen] = butter(2, plugin.HFCut/(getSampleRate(plugin)/2), 'low');
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
            wetDelayed = plugin.pFractionalDelay(wet, delaySamples);
                        
            % Mix the dry and wet signals together
            y = ((1-plugin.Mix) * u) + (plugin.Mix * wetDelayed);
        end

        function setupImpl(plugin, u)
            
            % Initialize reverb
            [RIRAudioLeft, RIRAudioRight] = getRIRAudio(plugin);
                        
            % Upsample to match input
            upsampledRIRAudioLeft = upsampleRIR(plugin, RIRAudioLeft);
            upsampledRIRAudioRight = upsampleRIR(plugin, RIRAudioRight);
            
            % Initialize HPF and LPF filters
            [plugin.HPFNum, plugin.HPFDen] = butter(2, plugin.LFCut/(getSampleRate(plugin)/2), 'high');
            [plugin.LPFNum, plugin.LPFDen] = butter(2, plugin.HFCut/(getSampleRate(plugin)/2), 'low');
            
            % Create frequency domain filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', upsampledRIRAudioLeft,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);

            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', upsampledRIRAudioRight,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFracDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
        end

        function resetImpl(plugin)
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);
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
            RIRIndex = (plugin.a * 100 + plugin.b * 10 + plugin.c) + 1;
            RIRAudioLeft = transpose(plugin.nverb.ir_audio(:,RIRIndex));
            RIRAudioRight = transpose(plugin.nverb.ir_audio(:,RIRIndex+1));
        end
        function upsampledRIRAudio = upsampleRIR(plugin, RIRAudio)
            upsampleFactor = floor(getSampleRate(plugin) / plugin.RIRFs);
            upsampledRIRAudio = upsample(RIRAudio, upsampleFactor);
        end
    end
    
    methods 
        function set.HFCut(plugin, val)
            plugin.HFCut = val;
            setUpdateLPF(plugin, true);
        end
        function val = get.HFCut(plugin)
            val = plugin.HFCut;
        end
        function set.LFCut(plugin, val)
            plugin.LFCut = val;
            setUpdateHPF(plugin, true);
        end
        function val = get.LFCut(plugin)
            val = plugin.LFCut;
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