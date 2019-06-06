%Wenrui Ye 
clc; clear;close all ;
jw = 0.2125 ;
jr = 10 ;
mw = 15 ;
mr = 85 ;
L = 1.2 ;
R = 0.3 ;
g = 9.81 ;
costheta = 1 ;
sintheta = 'theta' ;
s = zeros(2 , 2); %create a matrix for elements in EOM
s(1,1) = mr * L * costheta ;
s(1,2) = jr + mr * L^2 ;
s(2,1) = (2*jw)/(R^2) + mr + 2 * mw ;
s(2,2) = mr * L * costheta ;
a = mr * g * L ;  %a * sintheta - tau
b = mr * L^2 * 0 ; % b + tau/R
% ======================================================
%The stste space Variables ABCD and discrete system
A=[0 -18.74 0 ; 0 0 1; 0 21.99 0] ;
B=[0.0996 0 -0.0844]';
C=eye(3) ;
D=zeros(3,1);
Delta_t=0.05;
[Ad,Bd,Cd,Dd]=c2dm(A,B,C,D,Delta_t);
Q = eye(3);
Q(1,1) = 50 ;
R = 0.01 ;
x_initial = [-1 0 0]';
Co = ctrb(A,B);
Co_r = rank(ctrb(A,B));
Ob = obsv(A,C);
Ob_r = rank(obsv(A,C));
% =======================================================
% Generate Q R S T matrix
np = 22 ; % Set the prediction horizon and control horizon here
Q_size = size(Q) ;
Q_cell=cell(np,np);
for i=1:1:np
    for j=1:1:np
        if i==j
            Q_cell{i,j}=Q;
        else
            Q_cell{i,j}=zeros(Q_size(1),Q_size(2));
        end
    end
end

R_cell=cell(np,np);
R_size = size(R);
for i=1:1:np
    for j=1:1:np
        if i==j
            R_cell{i,j}=R;
        else
            R_cell{i,j}=zeros(R_size(1),R_size(2));
        end
    end
end
Q_bar = cell2mat(Q_cell) ;
R_bar = cell2mat(R_cell) ;

S_cell=cell(np,np);
S_size = size(Bd);
for i=1:1:np
    for j=1:1:np
        if i==j
            S_cell{i,j}=Bd;
         elseif i > j
            S_cell{i,j}=Ad^(i-j)*Bd;
         else
           S_cell{i,j}=zeros(3,1);
        end
    end
end
S_bar =cell2mat(S_cell) ;

T_cell=cell(np,1);
T_size = size(Ad);
for i=1:1:np
    T_cell{i,1}=Ad^i;
end
T_bar =cell2mat(T_cell) ;

% ===========================================================
% Calculate H & F based on the formula
H =2*(R_bar + S_bar'* Q_bar * S_bar);
F =(2*T_bar' * Q_bar * S_bar)';
Kmpc = inv(H) * F ;
timevector = 0:0.05:999*0.05;
n=length(timevector) ;
x=zeros(3 , 1000);
x(:,1) = x_initial ;
  for k= 1:n
      U = Kmpc*x(:,k);
      x(:,k+1) = Ad*x(:,k)-Bd*U(1,:) ;
  end
%   plot(x(1,:))
 circle = eig(Ad - Bd * Kmpc(1,:)) ;
 sqrt(circle(1)^2+circle(2)^2);
 sqrt(circle(1)^2+circle(3)^2);


   for i = 1:1:3
    subplot(1,3,i);
    plot(x(i,:))
    title(['For State ' , num2str(i)]);
    xlabel('Sample');
    ylabel('State Value');
    hold on
  end
