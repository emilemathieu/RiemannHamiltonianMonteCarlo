function PP = ModelParameterPrior(ParaNum, Value)

%%%%%%%%%
% Gamma %
%%%%%%%%%
if Value == 'random'
    % Produce random value from the prior
    PP = gamrnd(1, 3);
else
    if (Value < 0)
        PP = 0;
    else
        % Calculate probability of value from the prior
        PP = gampdf(Value, 1, 3);
    end
end

        
        
end

