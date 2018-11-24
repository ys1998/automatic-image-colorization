function [] = myMainScript()
%% Same input (RGB) and test (grayscale) image
knn_main('images/baboon', 'gray_images/gr_baboon.png');
pause(0.1);
knn_main('images/lena', 'gray_images/gr_lena.png');
pause(0.1);
knn_main('images/airplane', 'gray_images/gr_airplane.png');
pause(0.1);
knn_main('images/peppers', 'gray_images/gr_peppers.png');
pause(0.1);
knn_main('images/fruits', 'gray_images/gr_fruits.png');
pause(0.1);
%% Different input and test images
knn_main('images/trial', 'gray_images/gr_face.png');
pause(0.1);
%% Multiple training images
knn_main('images/trial2', 'gray_images/gr_face.png');
end