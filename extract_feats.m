function features = extract_feats(img, pixels, edge_length)
if(nargin <= 2)
    edge_length = 11;
end
% pixels - Mx2 dimensional matrix
% img - single channel image (matrix)
f1 = extract_surf_feats(img, pixels, edge_length);
f2 = extract_fft_feats(img, pixels, edge_length);
% f3 = extract_pos_feats(img, pixels, edge_length);
% features = [f1 f2 f3];
features = [f1 f2];
end

function features = extract_surf_feats(img, pixels, edge_length)
octave1 = img;
octave2 = imgaussfilt(img, 2);
octave3 = imgaussfilt(img, 4);
[f1, ~] = extractFeatures(octave1, pixels, 'Method', 'SURF', 'BlockSize', edge_length, 'FeatureSize', 128);
[f2, ~] = extractFeatures(octave2, pixels, 'Method', 'SURF', 'BlockSize', edge_length, 'FeatureSize', 128);
[f3, ~] = extractFeatures(octave3, pixels, 'Method', 'SURF', 'BlockSize', edge_length, 'FeatureSize', 128);
features = [f1 f2 f3];
end

function features = extract_pos_feats(img, pixels)
[H, W] = size(img);
features = [pixels(:, 1)./W, pixels(:, 2)./H];
end

function features = extract_fft_feats(img, pixels, edge_length)
features = zeros(size(pixels, 1), edge_length^2);
[H, W] = size(img);
cntr = 1;
for pos = pixels'
    if(not(pos(1) <= (edge_length-1)/2 || pos(1) > W-(edge_length-1)/2 || pos(2) <= (edge_length-1)/2 || pos(2) > H-(edge_length-1)/2))
        chunk = abs(fftshift(fft2(img(pos(1)-(edge_length-1)/2:pos(1)+(edge_length-1)/2, pos(2)-(edge_length-1)/2:pos(2)+(edge_length-1)/2))));
        features(cntr, :) = chunk(:)';
    end
    cntr = cntr + 1;
end
end
