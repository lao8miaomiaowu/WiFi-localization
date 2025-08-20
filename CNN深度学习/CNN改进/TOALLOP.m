function theta=TOALLOP(A,p,j)
%A是BBS的坐标
%P是范围测量
%J是参考BS的索引
[m,~]=size(A);  %size得到A的行列数赋值给[m,~]，~表示占位，就是只要行m的值
k=sum(A.^2,2);%矩阵A每个元素分别平方，得到新矩阵，在行求和，最为矩阵K
k1=k([1:j-1,j+1:m],:); %取出J行
A1=A([1:j-1,j+1:m],:); %取出J行
A2=A1-ones(m-1,1)*A(j,:); %得到D，就是j行与其余行对应值相减
p1=p([1:j-1,j+1:m],:); %取出J行
p2=p(j).^2*ones(m-1,1)-p1.^2-(k(j)*ones(m-1,1)-k1); %得到b,(Rn*Rn-R1*R1-Kn+K1)其中Kn为对应第n个x^2+y^2
theta=1/2*inv(A2'*A2)*A2'*p2; %利用最小二乘解，得位置估计
theta=theta';%转换为（x,y）形式
