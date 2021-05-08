BotGNN = dlmread('~/tdash/BotGNN/Results/BotGNN.csv',',',1,1);
BotGNN_AB = dlmread('~/tdash/BotGNN/Results/BotGNN_AB.csv',',',1,1);
GNN = dlmread('~/tdash/VEGNN/Results/GNN.csv',',',1,1);
VEGNN = dlmread('~/tdash/VEGNN/Results/VEGNN.csv',',',1,1);
DRM = dlmread('~/tdash/Basic_DRM/withBondInfo/Results/DRM.csv',',',1,1);
%DRM_abfr = dlmread('~/tdash/Basic_DRM/Results/DRMabfr.csv',',',1,1);
XGB = dlmread('~/tdash/Basic_XGB/XGBoutputs/results.csv',',',1,1);
SVM = dlmread('~/tdash/Basic_SVM/SVCoutputs/results.csv',',',1,1);
GPC = dlmread('~/tdash/Basic_GPC/GPCoutputs/results.csv',',',1,1);
CILP = dlmread('~/tdash/CILP/MLP_MultiHL/Results/CILP.csv',',',1,1);
CILPab = dlmread('~/tdash/CILP/MLP_MultiHL/Results/CILPab.csv',',',1,1);
ALEPH = dlmread('~/tdash/BotGNN/Results/aleph_minacc_0.7.csv',',',1,1);

for i = 1:5
    %A = GNN(:,i); B = BotGNN(:,i);
    %A = VEGNN(:,i); B = BotGNN(:,i);
    %A = DRM(:,1); B = BotGNN(:,i);
    %A = CILP(:,1); B = BotGNN(:,i);
    %A = SVM(:,1); B = BotGNN(:,i);
	%A = XGB(:,1); B = BotGNN(:,i);
    A = ALEPH(:,1); B = BotGNN(:,i);
    %A = round(A,5); B = round(B,5);
    
	gt = sum(B > A);
    lt = sum(B < A);
    eq = sum(B == A);
    [p, h] = signrank(A,B);
    gain = (median(B ./ A)-1)*100.0;
    fprintf('%d/%d/%d (%5.4f)\t median gain: %d%%\n',gt,lt,eq,p,round(gain));
    %disp([num2str(gt),'/',num2str(lt),'/',num2str(eq),' (',num2str(p),')','    [median gain:',num2str(gain),'%]']);
end
