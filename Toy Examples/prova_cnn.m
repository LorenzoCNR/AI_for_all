

%Input_=normrnd(0,1,100,8)
Kernel=rand(3,8,8)
Input_=normrnd(0,1,8,80)
stride=1

%%% Genero i bias da una Weibull
pd = makedist('Weibull');
rng('default') ; 
bias = random(pd,8,1);


N_L = custom_conv1d(Input_, Kernel, bias, stride, 'same', 'gelu');
