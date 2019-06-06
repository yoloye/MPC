addpath('C:\Users\anton\Desktop\casadi')
import casadi.*
clc; clear all;
close all;

% =====================================================================
% Set up the system
% =====================================================================
Time_period =15 ; %Total simulate time
T = 0.1; %sample time(s)
N = 20; %This is Prediction Horizon, this value needs to change!!!!!

%Assign values to variables, based on specific car model
C_af = 30000;%Coenering stiffness of front tires
C_ar = 58000;%Coenering stiffness of rear tires
L_f = 1.43;%Longitudinal distance from center of gravity to front tires
L_r = 1.595;%Longitudinal distance from center of gravity to rear tires
m = 1925;%The mass of vehicle
V_x = 15;%Longitudinal Velocity at center of gravity of vechicle
I_z = 3523;%Yaw moment of inertia of vehicle
% set up state space
%%input: Front steering angle
%%output: Global Y position
%state: [y_dot; yaw angle ; yadot]; 

A_1 = [ -2*(C_af + C_ar)/(m * V_x) 0 -V_x-2*(C_af*L_f + C_ar*L_r)/(m * V_x)];
A_2 = [ 0 0 1];
A_3 = [ -2*(C_af*L_f - C_ar*L_r)/(I_z * V_x) 0 -2*(C_af*L_f^2 + C_ar*L_r^2)/(I_z * V_x)];
A = [A_1; A_2 ; A_3]; 
B = [2*C_af/m ; 0 ; 2*L_f*C_af/I_z];
%output[Lateral position ; yaw angle]
C = [1 V_x 0];
D = 0;
Delta_t = T;
[Ad,Bd,Cd,Dd]=c2dm(A,B,C,D,Delta_t);
[Ad_row, Ad_col] = size(Ad);
Co = ctrb(A,B);
Co_r = rank(ctrb(A,B));
Ob = obsv(A,C);
Ob_r = rank(obsv(A,C));
% =====================================================================
% Set up the Control problem with casadi
% =====================================================================
% Set up states of system model
y = SX.sym('y'); y_dot = SX.sym('y_dot');
phi = SX.sym('phi'); phi_dot = SX.sym('phi_dot');
states = [y_dot; phi; phi_dot];
n_states = length(states);

% Set up control(input) of system model
alpha = SX.sym('alpha');
controls = [alpha];
n_controls = length(controls);

rhs = Ad*states+Bd*controls;
% rhs is right hand side equation


% =====================================================================
% Transfer OCP to NLP
% =====================================================================

f = Function('f', {states, controls}, {rhs});
% Define a function "f", it will take states and controls as variable
% and return rhs
U = SX.sym('U', n_controls, N); %Store information of Decision Variables(controls)
p = SX.sym('p', n_states + n_states);
% Parameters 'p', this part of this matrix(Length(n_states)) store 
% information of initial state of x(x0), and second part of it
% contains information of the reference state.
X = SX.sym('x', n_states,(N+1));
% A vector that contains states over the pridiction horizon period
% and the N+1 means that it contains the intial condition from last
% control period.
obj = 0 ; %objective function, will be continous update
g = []; %constrain vector
Q = zeros(n_states, n_states); %Weighting matrix of states
Q(1,1) = 1; Q(2,2) = 5; Q(3,3) = 0.1 ;
%Adjust this weight matrix to achieve optimal control
R = zeros(n_controls, n_controls); %Weighting matrix of controls
R(1,1) = 0.5;
%Adjust this weight matrix to achieve optimal control
st = X(:, 1); 
%X(:, 1) initial state, and this value will be updated in the following loop
g = [g; st - p(1:n_states)];
% initial condition constraints

% The following loop is aim to calculate the cost function and constrains
for i = 1:N
    st = X(:, i); 
    con = U(:, i);
    obj = obj + (st - p((n_states + 1):(n_states * 2)))' * Q *...
        (st - p((n_states + 1):(n_states * 2))) + con' * R * con;
    % 'obj' is the cost function, and this loop will be calculate the sum of
    % cost over the prediction horizon,(st - p) is the difference between
    % actual state and the reference, then times the penalize term to get
    % the cost. the objective is to minimize the cost
    st_next = X(:, i + 1);
    f_value = f(st, con); %this is define in previous step, it will return rhs
    %st_next_predict = st + (T * f_value);
    st_next_predict = f_value; % you already defined a discrete time model
    % the state of next time step
    g = [g; st_next - st_next_predict];
    %paraller compute the constrains, it equals 
    %state_next - state_next_predict
end

% Form the NLP 
OPT_variables = [reshape(X, n_states * (N+1),1); reshape(U, n_controls * N,1)];
% reshape all predict states and controls variable to a one column vector

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', p);
opts = struct;
opts.ipopt.max_iter = 2000;
% the algorithm will terminate with an error after max_iter times
% iterations
opts.ipopt.print_level = 0;
% The larger the value, the more detail in output. [0 12]
opts.print_time = 0;
% A boolean value. print informatin of execution time
opts.ipopt.acceptable_tol = 1e-8;
% The convergence tolerance
opts.ipopt.acceptable_obj_change_tol = 1e-6;
%Stop criterion
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% =====================================================================
% Set up the constrain of this OCP
% !!!!!!!!!!!This part needs to adjust later based on system
% =====================================================================
args = struct;

%Constrains of difference between actual value and predict value
%lb means lower bound, ub means upper bound
args.lbg(1: n_states * (N+1)) = 0;
args.ubg(1: n_states * (N+1)) = 0;

%following constrains are not based on the fact, need to sub with other
%values


%inf means no constrain
args.lbx(1: n_states: n_states * (N+1), 1) = -10;
args.ubx(1: n_states: n_states * (N+1), 1) = 10;
%Constrains of state y_dot
args.lbx(2: n_states: n_states * (N+1), 1) = -pi/6;
args.ubx(2: n_states: n_states * (N+1), 1) = pi/6;
%Constrains of state yaw angle(phi)
args.lbx(3: n_states: n_states * (N+1), 1) = -pi/12;
args.ubx(3: n_states: n_states * (N+1), 1) = pi/12;
%Constrains of state yaw angle dot(phi_dot)


%Constrains of control variable u(steering angle)
args.lbx(3*(N+1)+1: 1: n_states * (N+1) +n_controls * N) = -pi/3;
args.ubx(3*(N+1)+1: 1: n_states * (N+1) +n_controls * N) = pi/3;

% =====================================================================
% Set up the simulation loop
% =====================================================================
t0 = 0;
x0 = [0.0 ;0.0 ;0.0];% initialize states, based on number of states
xs = [0; 0.5; 0]; %!!!!!!!!!!!!!!!! Here needs to be changed
%This xs is the reference, the value of this should be determine after the
%path should be followed

x_state_history(:, 1) = x0;
t(1) = t0;
u0 = zeros(N, 1);
X0 = repmat(x0, 1, N+1)';
% refer as repeat matrix, a n+1 * 1 cell matrix , and each cell contains
% the matrix x0, the cell to mat
sim_time = Time_period;
mpciter = 0;
x_control = [];
u_control = [];
main_loop = tic;
while(norm((x0 - xs),2) > 1e-3 && mpciter < sim_time/T)
    % this condition is check the error
    args.p = [x0 ; xs];
    args.x0 = [reshape(X0',n_states * (N+1),1); reshape(u0', n_controls * N,1)];
    sol = solver('x0',args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    u = reshape(full(sol.x(n_states * (N+1)+1:end))', 1 , N)';
    x_control(:,1:n_states, mpciter+1) = reshape(full(sol.x(1:n_states * (N+1)))', n_states, N+1)';
    % this is a 3-D matrix, get controls from solution
    u_control = [u_control; u(1,:)];
    %This matrix save the information of control strategy
    %This is the most important information, use this to plot
    t(mpciter+1) = t0;
    %The current time, will update with literation
    [t0, x0, u0] = shift(T, t0, x0, u, f);
    x_state_history(:, mpciter + 2) = x0;
    %This matrix save the information of state
    %This is the most important information, use this to plot
    X0 = reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)';
    X0 = [X0(2:end,:); X0(end,:)];
    mpciter = mpciter + 1;
end

main_loop_time = toc(main_loop);
ss_error = norm((x0-xs),2)
average_mpc_time = main_loop_time/(mpciter+1)

subplot(1,3,1);
plot(x_state_history(1,:))
title(['State "ydot" ']);
xlabel('Sample');
ylabel('Velocity(m/s)');
yline(xs(1,1),'-.b');
legend('Actual State','Reference')
hold on

subplot(1,3,2);
plot(x_state_history(2,:))
title(['State "theta" ']);
xlabel('Sample');
ylabel('Angle(radius)');
yline(xs(2,1),'-.b');
legend('Actual State','Reference')
ylim([0 1.2*xs(2,1)])
hold on

subplot(1,3,3);
plot(x_state_history(3,:))
title(['State "thetadot" ']);
xlabel('Sample');
ylabel('Angular Velocity(radius/s)');
yline(xs(1,1),'-.b');
legend('Actual State','Reference')
ylim([-0.1 0.3])
hold off

function [t0, x0, u0] = shift(T, t0, x0, u,f)
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = f_value;
    x0 = full(st);
    t0 = t0 + T;
    u0 = [u(2:size(u,1),:); u(size(u,1), :)];
% Update u0, u from this step is the u0 of next step
% This has to be change if there are more than one input
end
