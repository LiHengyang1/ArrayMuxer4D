function mappingSpace = Fx_PredictionMAT(imagetargetPY)

imagetargetPY = single(imagetargetPY);
imagetargetPY = py.numpy.array(imagetargetPY);
mappingSpace = py.Fx_PredictionPY.pyfun0(imagetargetPY);
mappingSpace = double(mappingSpace);
