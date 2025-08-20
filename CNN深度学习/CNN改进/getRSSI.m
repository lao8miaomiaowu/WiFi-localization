function rssiValue=getRSSI(d,Q1,M)
A=-40;
n=2.2;
rssiValue=ones(1,M)*(A-10*n*log10(d));
% rssiValue=rssiValue+sqrt(Q1)*randn;
rssiValue = noisegen(rssiValue,Q1);