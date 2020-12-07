BotGNN = dlmread('~/tdash/BotGNN/Results/BotGNN.csv',',',1,1);
BotGNN_AB = dlmread('~/tdash/BotGNN/Results/BotGNN_AB.csv',',',1,1);
GNN = dlmread('~/tdash/VEGNN/Results/GNN.csv',',',1,1);
VEGNN = dlmread('~/tdash/VEGNN/Results/VEGNN.csv',',',1,1);
DRM = load('~/tdash/Basic_DRM/Results/DRM.csv');
DRM_abfr = load('~/tdash/Basic_DRM/Results/DRM_abfr.csv');

for i = 1:5
    %A = BotGNN_AB(:,i); B = BotGNN(:,i);
    A = DRM_abfr(:,1); B = DRM(:,1);
    %A = round(A,4); B = round(B,4);
    
    gt = sum(B > A);
    lt = sum(B < A);
    eq = sum(B == A);
    [p, h] = signrank(A,B);
    disp([num2str(gt),'/',num2str(lt),'/',num2str(eq),' (',num2str(p),')']);
end
