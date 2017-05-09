#!/bin/sh
#extract features for training
#python extractFeature.py models/apc40_final_iter_380000.caffemodel trainImages/ gtImages/ featuresLabels/

#training random forest
#python trainRF.py featuresLabels/feature20.txt featuresLabels/label20.txt models/

#python testRF.py testImages/all-20.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
#test image
python testRF.py trainImages/1-10.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
exit
python testRF.py testImages/1-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/2-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/3-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/4-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/5-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/6-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/7-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/8-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/9-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/10-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/11-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/12-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/13-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/14-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/15-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/16-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/17-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/18-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/19-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
python testRF.py testImages/20-simple.jpg gtImages/ models/apc40.rf.model models/apc40_final_iter_380000.caffemodel
