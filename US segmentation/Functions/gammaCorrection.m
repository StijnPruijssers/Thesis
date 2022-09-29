function [img_gamma] = gammaCorrection(img, gamma)

    invGamma = 1/gamma;
    table = [0:255];

    for i=1:size(table, 2)
        table(i) = ((table(i)/255)^invGamma) * 255;
    end
    table = uint8(table);

    % Lookuptable functionality
    img_gamma = zeros(size(img), 'uint8');

    for i=0:255
        img_gamma(img==i) = table(i+1); 
    end

end