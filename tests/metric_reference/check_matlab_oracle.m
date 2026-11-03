function check_matlab_oracle(folder)
% Cross-language synthetic-unit oracle. No dataset or model is loaded.
% Retains and calls pinned author source; never uses analysis_Reference wrapper.
here = fileparts(mfilename('fullpath'));
assert(~isfile(fullfile(folder,'matlab_oracle.csv')),'Refusing to overwrite existing MATLAB oracle');
addpath(fullfile(here,'matlab'));
addpath(fullfile(here,'fmi'));
data = load(fullfile(folder,'synthetic_inputs.mat'));
A=double(data.A); B=double(data.B); F=double(data.F);
names={'EN','MI','FMI','PSNR','Qabf','Nabf','AG','SF','SD','CC','SCD','SSIM','MS_SSIM','VIF'};
v=zeros(1,14);
v(1)=entropy(uint8(F));
% Explicit double histogram indices avoid uint8+1 saturation in Hab.m.
mi=@(X,Y) joint_mi(X,Y);
v(2)=mi(A,F)+mi(B,F);
v(3)=fmi(A,B,F,'dct',3);
mse=.5*mean((F-A).^2,'all')+.5*mean((F-B).^2,'all');
v(4)=10*log10(255^2/mse);
v(5)=xp_paper(A,B,F);
% Correct parameter order: analysis_nabf(fused,sourceA,sourceB).
v(6)=analysis_nabf(F,A,B);
v(7)=AG_evaluation(F);v(8)=SF_evaluation(F);v(9)=SD_evaluation(F);
v(10)=.5*(corr2(A,F)+corr2(B,F));v(11)=analysis_SCD(A,B,F);
v(12)=.5*(ssim_index(A,F)+ssim_index(B,F));
v(13)=.5*(msssim(A,F)+msssim(B,F));
v(14)=.5*(vifp_mscale(A,F)+vifp_mscale(B,F));
T=array2table(v,'VariableNames',names);writetable(T,fullfile(folder,'matlab_oracle.csv'));
expected=double(data.expected);
tol=[1e-10,1e-9,1e-7,1e-10,1e-9,1e-9,1e-10,1e-10,1e-10,1e-10,1e-10,1e-9,1e-8,1e-8];
assert(all(abs(v-expected)<=tol),'Cross-language mismatch: inspect matlab_oracle.csv');
fid=fopen(fullfile(folder,'MATLAB_PARITY_PASSED.txt'),'w');
fprintf(fid,'All 14 synthetic metrics passed against pinned MATLAB sources/formulas. MATLAB %s\n',version);
fclose(fid);
fprintf('MATLAB paper/source oracle passed for all 14 metrics.\n');
end

function val=joint_mi(X,Y)
p=accumarray([X(:)+1,Y(:)+1],1,[256,256]);p=p/sum(p,'all');
px=sum(p,2);py=sum(p,1);ind=px*py;use=p>0;
val=sum(p(use).*log2(p(use)./ind(use)));
end

function score=xp_paper(A,B,F)
% XP paper Eq (3),(4),(7),(8). Explicit ratio, circular angle and L=1.5;
% not Hong's known equal-magnitude branch or its L=1 compatibility setting.
[gf,af]=xp_edges(F);[ga,aa]=xp_edges(A);[gb,ab]=xp_edges(B);
wa=ga.^1.5;wb=gb.^1.5;
score=sum(xp_quality(ga,aa,gf,af).*wa+xp_quality(gb,ab,gf,af).*wb,'all')/sum(wa+wb,'all');
end

function [g,a]=xp_edges(X)
kx=[-1,0,1;-2,0,2;-1,0,1];ky=[1,2,1;0,0,0;-1,-2,-1];
x=conv2(X,kx,'same');y=conv2(X,ky,'same');g=sqrt(x.^2+y.^2);
a=pi/2*ones(size(X));use=x~=0;a(use)=atan(y(use)./x(use));
end

function q=xp_quality(g,a,gf,af)
d=max(g,gf);r=zeros(size(d));use=d>0;r(use)=min(g(use),gf(use))./d(use);
angle=abs(abs(a-af)-pi/2)/(pi/2);
q=.9994./(1+exp(-15*(r-.5))).*.9879./(1+exp(-22*(angle-.8)));
end
