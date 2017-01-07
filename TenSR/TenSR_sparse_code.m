function Sten = TenSR_sparse_code(Xten, Dten_cell, lambda, iteration)
% Compute sparse codes
%
%
%
%

dictionary_num = length(Dten_cell);

% Initialize
Sten_size = zeros(1,dictionary_num);
for i = 1:dictionary_num
    Sten_size(i) = size(Dten_cell{i},2);
end
Sten = cell(1,iteration);
Sten{1} = tensor(single(rand(Sten_size)));
Sten0 = Sten{1};
Cten = Sten;

Dten_cell_T = cell(1,dictionary_num);
for i = 1:dictionary_num
    Dten_cell_T{i} = Dten_cell{i}';
end

Dten_cell_T_Dten_cell = cell(1,dictionary_num);
for i = 1:dictionary_num
    Dten_cell_T_Dten_cell{i} = Dten_cell_T{i} * Dten_cell{i};
end


% Calculate Sten
L = 1;
t = ones(1,iteration);
for k = 1:iteration
    for i = 1:dictionary_num
        temp = Dten_cell{i}' * Dten_cell{i};
        L = L * sum(temp(:).^2);
    end
    L = eta^k * L;
    
    delta = ttm(Xten, [Dten_cell_T{:}], 1:dictionary_num) -...
            ttm(Cten, [Dten_cell_T_Dten_cell{:}], 1:dictionary_num);
    Sten{k} = tensor(sign(double(Cten - delta/L)) * max(abs(double(Cten - delta/L))-lambda/L,0));
    t(k+1) = (1+sart(1+4*t(k)^2))/2;
    Cten = Sten{k} + ((t(k)-1)/t(k+1)) * (Sten{k}-Sten{k-1});
end
end
