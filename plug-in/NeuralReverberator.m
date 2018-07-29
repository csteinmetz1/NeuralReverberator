classdef NeuralReverberator < audioPlugin & matlab.System & ...
        matlab.system.mixin.Propagates
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        nverb = load('/Volumes/HDMETZ1/Datasets/nverb/pre_compute/nverb.mat')
        ImpulseResponseFs = 16000;
        ImpulseResponseLength = 65280;
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
            'InputChannels',1,...
            'OutputChannels',1,...
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
        pFIR
        pFractionalDelay
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            
            ir_index = (plugin.a * 100 + plugin.b * 10 + plugin.c) + 1;
            ir_audio = transpose(plugin.nverb.ir_audio(:,ir_index));
            plugin.pFIR.Numerator = ir_audio;
            
            u = 10.^(plugin.InputGain/20)*u;
            wet = step(plugin.pFIR, u);
            dly_samples = (plugin.Pre_delay/1000) * getSampleRate(plugin);
            dly_wet = plugin.pFractionalDelay(wet, dly_samples);
            y = ((1-plugin.Mix) * u) + (plugin.Mix * dly_wet);
        end

        function setupImpl(plugin, u)
            
            % initialize reverb
            ir_index = (plugin.a * 100 + plugin.b * 10 + plugin.c) + 1;
            ir_audio = transpose(plugin.nverb.ir_audio(:,ir_index));
            
            % Create frequency domain filter for convolution 
            plugin.pFIR = dsp.FrequencyDomainFIRFilter('Numerator', ir_audio,...
                'PartitionForReducedLatency', true, 'PartitionLength', plugin.PartitionSize);
            
            % Create fractional delay
            plugin.pFractionalDelay = dsp.VariableFractionalDelay(...
                'MaximumDelay',192000*.3);
        end

        function resetImpl(plugin)
            reset(plugin.pFIR);
        end
        %------------------------------------------------------------------
        % Propagators
        function varargout = isOutputComplexImpl(~)
            varargout{1} = false;
        end
        
        function varargout = getOutputSizeImpl(obj)
            varargout{1} = propagatedInputSize(obj, 1);
        end
        
        function varargout = getOutputDataTypeImpl(obj)
            varargout{1} = propagatedInputDataType(obj, 1);
        end
        
        function varargout = isOutputFixedSizeImpl(obj)
            varargout{1} = propagatedInputFixedSize(obj,1);
        end
    end  
    
    methods 
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