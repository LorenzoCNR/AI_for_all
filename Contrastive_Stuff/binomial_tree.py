# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 02:15:01 2024

"""

import numpy as np

def binomial_tree(n, r, q, K, option_type, S0, sigma):
       '''
     build a binomial tree of the underlying asset evolution 
     check the option value and suggest if keep the option or exercise it
     
     
    Parameters
    for definitions see Hull 2021, ch 21 ( basic numerical procedures)
    ----------
    n :  integer (# of nodes of the tree...time intervals)
    r :  float (risk free interest rate)
    q :  float yield
    K : float strike price
    option_type : str  call or put
    S0 : float asset initial price
    sigma : float volatility

    Returns
    
    option value at 0
    -------
       '''
       
       ### the higher the time fractioning, the smaller the interval     
       dt=1/n
       ## up scaling factor
       u= np.exp(sigma * np.sqrt(dt))
       ## down scaling factor
       d = 1 / u
       # growth factor
       a = np.exp((r - q) * dt)
       # probability up
       p = (a - d) / (u - d)
       
       #à# initialize stock price tree
       stock_price_tree=np.zeros((n+1,n+1))
       ### loop to compute every possible price
       for i in range(n+1):
           for j in range(n+1):
               stock_price_tree[j,i]=S0*(u**(i-j))*(d**j)
               
        #### for a call 
       option_value_tree = np.zeros((n+1, n+1))
       if option_type.lower() == 'call':
            option_value_tree[:, n] = np.maximum(stock_price_tree[:, n] - K, 0)
       elif option_type.lower() == 'put':
            option_value_tree[:, n] = np.maximum(K - stock_price_tree[:, n], 0)
    
       for i in range(n-1, -1, -1):
            for j in range(i+1):
                ### compute exercise and intrinsic values (given the option type)
                continue_value = np.exp(-r * dt) * (p * option_value_tree[j, i+1] + (1 - p) * option_value_tree[j+1, i+1])
                if option_type.lower() == 'call':
                    exercise_value = stock_price_tree[j, i] - K
                elif option_type.lower() == 'put':
                    exercise_value = K - stock_price_tree[j, i]
                
                
                print(f"Node ({j}, {i}): Stock price = {stock_price_tree[j, i]:.2f}, "
                    f"hold value = {continue_value:.2f}, exercise = {exercise_value:.2f}")
                
                ###compare exercise and hold value 
                
                if exercise_value > continue_value:
                    option_value_tree[j, i] = exercise_value
                    print(f"    Exercise option at node({j}, {i})")
                else:
                    option_value_tree[j, i] = continue_value
                    print(f"    keep the option at node ({j}, {i})")

             #g   option_value_tree[j, i] = max(continue_value, exercise_value)

       return option_value_tree[0, 0]

def main():
    #### assign value to the parameters
    # nodes 
    n = 2 
    ### interest rate 
    r = 0.1  
    dividend_yield = 0.01
    ### strike      
    K = 50  
    option_type = 'put'  
    ##à initial asset price
    S0 = 50
    ### volatiliy
    sigma = 0.4

    option_value = binomial_tree(n, r, dividend_yield, K, option_type, S0, sigma)
    print("Option value is:", option_value)

if __name__ == "__main__":
    main()
        
               
        
        
        