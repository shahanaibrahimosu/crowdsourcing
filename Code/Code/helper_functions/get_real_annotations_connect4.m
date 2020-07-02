function [f,F,TestSet] = get_real_annotations_connect4(Data,K,M)



[TrainSet,TestSet]=getRandomTrainSet(Data);
N=size(TestSet,1);
[Model1, validationAccuracy1]=trainClassifierFineTreeConnect4(TrainSet);
%TrainSet=getRandomTrainSet(Data);
[Model2, validationAccuracy2]=trainClassifierMedTreeConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
[Model3, validationAccuracy3]=trainClassifierCoarseTreeConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
[Model4, validationAccuracy4]=trainClassifierCoarsekNNConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
 [Model5, validationAccuracy5]=trainClassifierFinekNNConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
 [Model6, validationAccuracy6]=trainClassifierMedkNNConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
[Model7, validationAccuracy7]=trainClassifierCosinekNNConnect4(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
% [Model8, validationAccuracy8]=trainClassifierLogRegression(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
% [Model9, validationAccuracy9]=trainClassifierLinearDiscrim(TrainSet);
% %TrainSet=getRandomTrainSet(Data);
% [Model10, validationAccuracy10]=trainClassifierQuadraticDiscrim(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model8, validationAccuracy8]=trainClassifierWeightedkNN(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model9, validationAccuracy9]=trainClassifierCoarseGaussianSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model10, validationAccuracy10]=trainClassifierMediumGaussianSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model11, validationAccuracy11]=trainClassifierLinearSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model12, validationAccuracy12]=trainClassifierQuadraticSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model13, validationAccuracy13]=trainClassifierCubicSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model14, validationAccuracy14]=trainClassifierFineGaussianSVM(TrainSet);
% TrainSet=getRandomTrainSet(Data);
% [Model15, validationAccuracy15]=trainClassifierCubickNN(TrainSet);
Data = TestSet;
f = zeros(M,N); %annotator labels
f(1,:)=Model1.predictFcn(Data(:,2:end));
f(2,:)=Model2.predictFcn(Data(:,2:end));
f(3,:)=Model3.predictFcn(Data(:,2:end));
f(4,:)=Model4.predictFcn(Data(:,2:end));
f(5,:)=Model5.predictFcn(Data(:,2:end));
f(6,:)=Model6.predictFcn(Data(:,2:end));
f(7,:)=Model7.predictFcn(Data(:,2:end));
% f(8,:)=Model8.predictFcn(Data(:,2:end));
% f(9,:)=Model9.predictFcn(Data(:,2:end));
% f(10,:)=Model10.predictFcn(Data(:,2:end));
% f(11,:)=Model11.predictFcn(Data(:,2:end));
% f(12,:)=Model12.predictFcn(Data(:,2:end));
% f(13,:)=Model13.predictFcn(Data(:,2:end));
% f(14,:)=Model14.predictFcn(Data(:,2:end));
% f(15,:)=Model15.predictFcn(Data(:,2:end));

F = cell(M,1); %cell of annotator responses. 
for i=1:M 
    indx = find(f(i,:) > 0);
    %N_i = numel(indx);
    F{i} = sparse(f(i,indx),indx,1,K,N);
end

end

function [TrainSet,TestSet] = getRandomTrainSet(Data)
[rows, col] = size(Data);
[trainInd,testInd] = dividerand(rows,0.7,0.3);
Data_train   = Data(trainInd,2:end);
Data_test    = Data(testInd,2:end);
Labels_train = Data(trainInd,1);
Labels_test  = Data(testInd,1);
TrainSet      = [Labels_train Data_train];
TestSet       = [Labels_test  Data_test];
end