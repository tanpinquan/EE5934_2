frameId  = 00003;
% frameId  = 16116;

% make filenames from frame id
imageFilename = sprintf('../data/GTA_V/train_img/%05d.png', frameId);
labelFilename = sprintf('../data/GTA_V/train_label/%05d.png', frameId);

img = imread(imageFilename);

% the labels are stored as class ids with a color mapping.
% to get the current color mapping from a file use
[labels, currentMapping] = imread(labelFilename);

% load mapping for CamVid and CityScapes
mapping = load('gta_mapping.mat', 'cityscapesMap', 'camvidMap', 'classes');

figure; 
imshow(img);
title(sprintf('Image %d', frameId));

figure; 
imshow(labels, mapping.cityscapesMap);
title('Labels (CamVid colors)');

figure; 
imshow(labels, mapping.camvidMap);
title('Labels (CamVid colors)');
