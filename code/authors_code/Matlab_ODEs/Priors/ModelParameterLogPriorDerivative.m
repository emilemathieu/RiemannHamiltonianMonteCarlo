function PP = ModelParameterLogPriorDerivative(ParaNum, Value)

% Return the partial derivative of the log(prior) w.r.t. the parameter

% Gamma Prior
a = 1;
b = 3;

if (Value < 0)
    PP = -inf;
else
    PP = (a-1)/Value - (1/b);
end


end
