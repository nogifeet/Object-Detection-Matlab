function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = repelem(X,layer.UpSampleFactor,layer.UpSampleFactor);
   end
  


