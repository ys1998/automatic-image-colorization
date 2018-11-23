function features = extract_feats(img, pixels)
% pixels - Mx2 dimensional matrix
% img - single channel image (matrix)
f1 = extract_surf_feats(img, pixels);
f2 = extract_fft_feats(img, pixels);
f3 = extract_pos_feats(img, pixels);
features = [f1 f2 f3];
end

function features = extract_surf_feats(img, pixels)
octave1 = img;
octave2 = imgaussfilt(img, 2);
octave3 = imgaussfilt(img, 3);
[f1, ~] = extractFeatures(octave1, pixels, 'Method', 'SURF', 'BlockSize', 21, 'FeatureSize', 128);
[f2, ~] = extractFeatures(octave2, pixels, 'Method', 'SURF', 'BlockSize', 21, 'FeatureSize', 128);
[f3, ~] = extractFeatures(octave3, pixels, 'Method', 'SURF', 'BlockSize', 21, 'FeatureSize', 128);
features = [f1 f2 f3];
end

function features = extract_pos_feats(img, pixels)
[H, W] = size(img);
features = [pixels(:, 1)./W, pixels(:, 2)./H];
end

function features = extract_fft_feats(img, pixels)
features = zeros(size(pixels, 1), 21^2);
[H, W] = size(img);
cntr = 1;
for pos = pixels'
    if(not(pos(1) <= 10 || pos(1) > W-10 || pos(2) <= 10 || pos(2) > H-10))
        chunk = abs(fftshift(fft2(img(pos(1)-10:pos(1)+10, pos(2)-10:pos(2)+10))));
        features(cntr, :) = chunk(:)';
    end
    cntr = cntr + 1;
end
end
