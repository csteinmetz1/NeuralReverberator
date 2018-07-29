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
        InputGain = -12;
        HF_cut = 20000.0;
        LF_cut = 1.0;
        Pre_delay = 0.0;
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
            audioPluginParameter('HF_cut','DisplayName','HF Cut','Label','Hz','Mapping',{'log',5000.0,20000.0}),...
            audioPluginParameter('LF_cut','DisplayName','LF Cut','Label','Hz','Mapping',{'log',1.0,5000.0}),...
            audioPluginParameter('Pre_delay','DisplayName','Pre-Delay','Label','ms','Mapping',{'lin',0.0,300.0}),...
            audioPluginParameter('a','DisplayName','A','Mapping',{'int',0,9}),...
            audioPluginParameter('b','DisplayName','B','Mapping',{'int',0,9}),...
            audioPluginParameter('c','DisplayName','C','Mapping',{'int',0,9}),...
            audioPluginParameter('Mix','DisplayName','Mix','Label','%','Mapping',{'lin',0,1.0}));
    end
    %----------------------------------------------------------------------
    % Private properties
    %----------------------------------------------------------------------
    properties(Access = private)
        pFIRLeft;
        pFIRRight;
        pFractionalDelay
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            
            % Initialize reverb
            [RIRAudioLeft, RIRAudioRight] = plugin.getRIR();
            
            % Upsample to match input
            upsampledRIRAudioLeft = plugin.upsampleRIR(RIRAudioLeft);
            upsampledRIRAudioRight = plugin.upsampleRIR(RIRAudioRight);
            
            % Update FIR filter
            plugin.pFIRLeft.Numerator = upsampledRIRAudioLeft;
            plugin.pFIRRight.Numerator = upsampledRIRAudioRight;
            
            u = 10.^(plugin.InputGain/20)*u;
            
            wetLeft = step(plugin.pFIRLeft, u(:,1));
            wetRight = step(plugin.pFIRRight, u(:,2));
            
            delaySamples = (plugin.Pre_delay/1000) * getSampleRate(plugin);
            wetLeftDelayed = plugin.pFractionalDelay(wetLeft, delaySamples);
            wetRightDelayed = plugin.pFractionalDelay(wetRight, delaySamples);
            
            wetStereoDelayed = [wetLeftDelayed, wetRightDelayed];
            
            y = ((1-plugin.Mix) * u) + (plugin.Mix * wetStereoDelayed);
        end

        function setupImpl(plugin, u)
            
            % Initialize reverb
            [RIRAudioLeft, RIRAudioRight] = plugin.getRIR();
                        
            % Upsample to match input
            upsampledRIRAudioLeft = plugin.upsampleRIR(RIRAudioLeft);
            upsampledRIRAudioRight = plugin.upsampleRIR(RIRAudioRight);
            
            % Create frequency domain filters for convolution 
            plugin.pFIRLeft = dsp.FrequencyDomainFIRFilter('Numerator', upsampledRIRAudioLeft,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);

            plugin.pFIRRight = dsp.FrequencyDomainFIRFilter('Numerator', upsampledRIRAudioRight,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFractionalDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
        end

        function resetImpl(plugin)
            reset(plugin.pFIRLeft);
            reset(plugin.pFIRRight);
        end
    end
    methods 
        function [RIRAudioLeft, RIRAudioRight] = getRIR(plugin)
            RIRIndex = (plugin.a * 100 + plugin.b * 10 + plugin.c) + 1;
            RIRAudioLeft = transpose(plugin.nverb.ir_audio(:,RIRIndex));
            RIRAudioRight = transpose(plugin.nverb.ir_audio(:,RIRIndex+1));
        end
        function upsampledRIRAudio = upsampleRIR(plugin, RIRAudio)
            upsampleFactor = floor(getSampleRate(plugin) / plugin.RIRFs);
            upsampledRIRAudio = upsample(RIRAudio, upsampleFactor);
        end
        function set.a(plugin, val)
            plugin.a = val;
        end
        function val = get.a(plugin)
            val = plugin.a;
        end
        function set.b(plugin, val)
            plugin.b = val;
        end
        function val = get.b(plugin)
            val = plugin.b;
        end      
        function set.c(plugin, val)
            plugin.c = val;
        end
        function val = get.c(plugin)
            val = plugin.c;
        end
    end  
end