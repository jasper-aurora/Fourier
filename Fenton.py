# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:44:24 2023

@author: Aurora2
"""

import math
import numpy as np


class Fenton:
    
    def __init__(self, Tmax, Hmax, Depth, Uc, z_ref, D, Option, blimit):
        self.Tp = Tmax
        self.Hs = Hmax
        self.Depth = Depth
        self.Uc = Uc
        self.z_ref = z_ref
        self.y = D/2        # This is assuming cable is touching the seabed and marine growth is not growing under the cable
        self.Option = Option
        self.g = 9.81
        self.blimit = blimit
    
    def Fenton(self, T, H, d, Uc, yc, y, blimit, Option):
       # The main function to solve the non-linear equations based on Fenton's Fourier methond
       # •	Where does it come from: Rienecker M M, Fenton J D. A Fourier approximation method for steady water waves. J. Fluid Mech. 1981. 104, 119-137.
       # T         - Period of wave [s] 
       # H         - wave height [m]
       # d         - water depth [m]
       # Uc        - current veloity measrued at a specific elevation
       # yc        - position where the current velocity is measured [m]c
       # y         - position where the wave-induced velcotiy is to be determined [m]
       # Option    - 0 : local current velocty measured at yc is provided
       #           - 1 : depth-averaged current velocity is provide, in this option yc is set as default, yc = d
       # blimit    - breaking limit of wave H/d, assumed based on experience 
       # return    - Solution [eta0 eta B0 B c k Q R] and Residual error 'r_error'
       #             where, eta0, eta  - water surface along the wave length
       #                    B0, B      - Fourier coefficient
       #                    c          - group velcoity 
       #                    k          - wave nmuber
       #                    Q          - total volume rate of flow, and 
       #                    R          - a constant pressure
       # defaut values that could be defined as global variables
       N = 20    # - grid number along the wave length   
       # Non-dimensionalization of the input parameters
       # tau       - non-dimensional wave period
       # H1        - non-dimensional wave heihgt
       # cE        - non-dimensional current velocity: for option 0, cE = local, for option 1, cE = depth-averaged
       # Dc        - non-dimensional refence height
       tau=T*math.sqrt(self.g/d) 
       H1 = H/d
       
    #    if H1>blimit:                        # check the breaking limit
    #        H1 = blimit
       
       cc, Numit = self.down_hill(H1)            # call function 'down_hill' to calcualte the teration interval and step number
       cE = Uc/math.sqrt(self.g*d) 
       if Option == 1:
           yc = d
       Dc = yc/d                            # Reference height
       x = self.initial_guess (N, tau, H1)       # call function of 'initial_guess' to define the values of the zero-th iteration 
                                            # Here, x contains the initial guess of [eta0 eta B0 B c k Q R], based on Rienecker & Fenton (1981, the last 2 paragraphs on page 125) 
       for it in range(Numit):
           f = self.Fenton_f(N,H1,tau,cE,Dc,x,Option)        # call function 'Fenton_f' to calcualte the residual error after the it-th iterations
           r_error = np.sum(np.abs(f))
           if r_error < 10**(-13):                      # the limit smaller than which the calcualtion is finished
              break
           df = self.Fenton_df(N, H1, tau, cE, Dc,x,Option)  # call fucntion 'Fenton_df' to solve the Jaccobian matrix, based on pages 124-125 in Rienecker & Fenton (1981)
           temp = np.matrix(df)
           dfi = temp.I                                 # inverse of the matrix df
           temp = np.dot(dfi,f)
           for ii in range(46):                         # Newton iteration, the down-hill method with the coefficient of cc
               x[ii] = x[ii]-cc*temp[0,ii]
               
       Solution = x
       return Solution, r_error       
       
      
    def initial_guess (self, N, tau, H1):
        # a sub-function called by function 'Fenton', to define the values of the zero-th iteration 
        # •	Where does it come from: Rienecker & Fenton (1981, the last 2 paragraphs on page 125) 
        # return - x contains the initial guess of [eta0 eta B0 B c k Q R] 
       eta = np.zeros(N)
       B   = np.zeros(N)
       c = 1
       k=1
       for ii in range(10):  
            k=2*math.pi/(tau*c)
            c=(math.tanh(k)/k)**0.5
    
       eta0 = 1+0.5*H1*math.cos(0*math.pi/N); 
       for ii in range(N):
           eta[ii]  = 1+0.5*H1*math.cos((ii+1)*math.pi/N); 
       B0   = -c;
       B[0] =  -0.25*H1/c/k;
       R    =  1+0.5*c**2;
       Q    =  c;
       x= np.append(eta0, eta)
       x= np.append(x, B0)
       x= np.append(x, B)
       x= np.append(x, c)
       x= np.append(x, k)
       x= np.append(x, Q)
       x= np.append(x, R)
       return x
    
    def Fenton_f(self, N,H1,tau,cE,Dc,x,Option):
        # a sub-function called by the function 'Fenton', to calcualte the resiudal error after each iteration
        # •	Where does it come from: Rienecker & Fenton (1981, equation 8 to equation 13) 
        # return - f contains the residuals for each equation, number of equaiotns are 2*N+6
        eta0 = x[0]
        eta = np.zeros(N)
        B   = np.zeros(N)
        for ii in range (N):
            eta[ii] = x[ii+1]
        B0  = x[N+1]
        for ii in range(N):
            B[ii]   = x[N+2+ii]
        c   = x[2*N+2]
        k  =  x[2*N+3]
        Q  =  x[2*N+4]
        R  =  x[2*N+5]
        temp=0
        for j in range(N):
            temp=temp+B[j]*math.sinh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.cos((j+1)*0*math.pi/N)
        f80 = B0*eta0+temp+Q            # equation 8 for eta0 and B0
        f8 = np.zeros(N)                # equation 8 for eta1 to etaN and B1 to BN
        for m in range(N):
            temp=0
            for j in range(N):
                temp=temp+B[j]*math.sinh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.cos((j+1)*(m+1)*math.pi/N)
            f8[m]=B0*eta[m]+temp+Q
    
        temp=0
        for j in range(N):
            temp=temp+(j+1)*B[j]*math.cosh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.cos((j+1)*0*math.pi/N)
        u0=B0+k*temp
        
        temp=0
        for j in range(N):
            temp=temp+(j+1)*B[j]*math.sinh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.sin((j+1)*0*math.pi/N)
        v0=k*temp
        f90=0.5*u0**2+0.5*v0**2+eta0-R      # equation 9 for eta0 and B0
        
        u=np.zeros(N)
        v=np.zeros(N)
        f9 = np.zeros(N)                    # equation 9 for eta1 to etaN and B1 to BN
        for m in range(N):
            temp=0
            for j in range(N):
                temp=temp+(j+1)*B[j]*math.cosh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.cos((j+1)*(m+1)*math.pi/N)
            u[m]=B0+k*temp
        
            temp=0
            for j in range(N):
                temp=temp+(j+1)*B[j]*math.sinh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.sin((j+1)*(m+1)*math.pi/N)
            v[m]=k*temp
        
            f9[m]=0.5*u[m]**2+0.5*v[m]**2+eta[m]-R
    
        temp = 0
        for ii in range(0,N-1):
            temp = temp+eta[ii]
            
        f10=(eta0+eta[N-1]+2*temp)/2/N-1    # equation 10
        f11=eta0-eta[N-1]-H1                # equation 11
        f12=k*c*tau-2*math.pi               # equation 12
        if Option == 0:
            f13=c-cE+B0                     # Option 0, equation 13a
        elif Option == 1:
            f13 =c-cE-Q                     # Option 1, equation 13b
        f=np.append(f80, f8)
        f=np.append(f, f90)
        f=np.append(f, f9)
        f=np.append(f, f10)
        f=np.append(f, f11)
        f=np.append(f, f12)
        f=np.append(f, f13)
        return f
    
    def Fenton_df(self, N, H1, tau, cE, Dc,x,Option):
        # a sub-function called by the function 'Fenton', to calcualte the Jaccobian matrix of the non-linear euqations
        # •	Where does it come from: Rienecker & Fenton (1981, pages 124 and 125) 
        # return - df the Jaccobian matrix with dimension of (2N+6,2N+6)
        eta0 = x[0]
        eta = np.zeros(N)
        B   = np.zeros(N)
        for ii in range (N):
            eta[ii] = x[ii+1]
        B0  = x[N+1]
        for ii in range(N):
            B[ii]   = x[N+2+ii]
        c   = x[2*N+2]
        k  =  x[2*N+3]
        Q  =  x[2*N+4]
        R  =  x[2*N+5]
    
        temp=0
        for j in range(N):
            temp=temp+(j+1)*B[j]*math.cosh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.cos((j+1)*0*math.pi/N)
        u0=B0+k*temp
        
        temp=0
        for j in range(N):
            temp=temp+(j+1)*B[j]*math.sinh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.sin((j+1)*0*math.pi/N)
        v0=k*temp
          
        u=np.zeros(N)
        v=np.zeros(N)
        for m in range(N):
            temp=0
            for j in range(N):
                temp=temp+(j+1)*B[j]*math.cosh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.cos((j+1)*(m+1)*math.pi/N)
            u[m]=B0+k*temp
        
            temp=0
            for j in range(N):
                temp=temp+(j+1)*B[j]*math.sinh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.sin((j+1)*(m+1)*math.pi/N)
            v[m]=k*temp
        
    
        df = np.zeros((2*N+6,2*N+6))
        i = 0
        df[i,i] = u0                                                                # dfi/deta0
        df[i,N+1] = eta0                                                            # dfi/dB0   % not correct in Rienecker & Fenton (1981)
        for j in range (N):
            df[i,N+2+j]= self.SJM10(j,0,k,eta0,Dc,N)                                     # dfi/dB
        temp=0
        for j in range(N):
            temp=temp+(j+1)*B[j]*self.SJM10(j,0,k,eta0,Dc,N)*math.tanh((j+1)*k*Dc)
    
        df[i,2*N+3] = eta0*(u0-B0)/k-Dc*temp                                        # dfi/dk
        df[i,2*N+4] = 1                                                             # dfi/dQ
        
    
        
        for ii in range(N):
            i = ii+1
            m = i-1
            df[i,i] =  u[m]                                                         # dfi/deta 
            df[i,N+1] = eta[m]                                                      # dfi/dB0   % not correct in Rienecker & Fenton (1981)
            for j in range(N):
                df[i,N+2+j]  = self.SJM1(j,m,k,eta,Dc,N)                                 # dfi/dB
            temp=0
            for j in range(N):
                temp = temp+(j+1)*B[j]*self.SJM1(j,m,k,eta,Dc,N)*math.tanh((j+1)*k*Dc)
            df[i,2*N+3] = eta[m]*(u[m]-B0)/k-Dc*temp                                # dfi/dk
            df[i,2*N+4] = 1                                                         # dfi/dQ
    
    
        i = N+1
        m= i - (N+1)
        temp1=0
        for j in range(N):
            temp1=temp1+(j+1)**2*B[j]*self.SJM10(j,0,k,eta0,Dc,N)
    
        temp2=0
        for j in range(N):
            temp2=temp2+j**2*B[j]*self.CJM10(j,0,k,eta0,Dc,N)
    
        df[i,0] = 1+u0*k**2*temp1+v0*k**2*temp2                                     # dfi/deta0
        df[i,N+1] = u0                                                              # dfi/dB0   % not correct in Rienecker & Fenton (1981)
        for j in range(N):
            df[i,N+2+j] = (j+1)*k*u0*self.CJM20(j,m,k,eta0,Dc,N)+(j+1)*k*v0*self.SJM20(j,m,k,eta0,Dc,N)       # dfi/dB
    
    
        temp1=0
        for j in range(N):
            temp1=temp1+(j+1)**2*B[j]*self.SJM10(j,0,k,eta0,Dc,N)
        temp2=0
        for j in range(N):
            temp2=temp2+(j+1)**2*B[j]*self.CJM20(j,0,k,eta0,Dc,N)*math.tanh((j+1)*k*Dc)
        temp3=0
        for j in range(N):
            temp3=temp3+(j+1)**2*B[j]*self.CJM10(j,0,k,eta0,Dc,N)
        temp4=0
        for j in range(N):
            temp4=temp4+(j+1)**2*B[j]*self.SJM20(j,0,k,eta0,Dc,N)*math.tanh((j+1)*k*Dc)
    
        df[i,2*N+3] =  u0*((u0-B0)/k+k*eta0*temp1-k*Dc*temp2)+v0*(v0/k+k*eta0*temp3-k*Dc*temp4)     # dfi/dk
        df[i,2*N+5] = -1;                                                                           # dfi/dR
    
        for ii in range(N):
            i = N+2+ii
            m= i - (N+2)
            temp1=0
            for j in range(N):
                temp1=temp1+(j+1)**2*B[j]*self.SJM1(j,m,k,eta,Dc,N)
            temp2=0
            for j in range(N):
                temp2=temp2+(j+1)**2*B[j]*self.CJM1(j,m,k,eta,Dc,N)
            df[i,m+1] = 1+u[m]*k**2*temp1+v[m]*k**2*temp2                           # dfi/deta
            df[i,N+1] = u[m]                                                        # dfi/dB0   % not correct in Rienecker & Fenton (1981)
            for j in range(N):
                df[i,N+2+j]= (j+1)*k*u[m]*self.CJM2(j,m,k,eta,Dc,N)+(j+1)*k*v[m]*self.SJM2(j,m,k,eta,Dc,N)   # dfi/dB
        
        
            temp1=0
            for j in range(N):
                temp1=temp1+(j+1)**2*B[j]*self.SJM1(j,m,k,eta,Dc,N)
            temp2=0
            for j in range(N):
                temp2=temp2+(j+1)**2*B[j]*self.CJM2(j,m,k,eta,Dc,N)*math.tanh((j+1)*k*Dc)
            temp3=0
            for j in range(N):
                temp3=temp3+(j+1)**2*B[j]*self.CJM1(j,m,k,eta,Dc,N)    
            temp4=0
            for j in range(N):
                temp4=temp4+(j+1)**2*B[j]*self.SJM2(j,m,k,eta,Dc,N)*math.tanh((j+1)*k*Dc)
        
            df[i,2*N+3] = u[m]*((u[m]-B0)/k+k*eta[m]*temp1-k*Dc*temp2)+v[m]*(v[m]/k+k*eta[m]*temp3-k*Dc*temp4) # dfi/dk
            df[i,2*N+5] = -1                                                                                   # dfi/dR
    
        i = 2*N+2
        df[i,0] = 1/2/N                                                             #df/deta
        df[i,N]=1/2/N
        for jj in range(N-1):
            df[i,jj+1] = 1/N
        i = 2*N+3
        df[i,0]=1                                                                   
        df[i,N]=-1
        i=2*N+4
        df[i,2*N+2] = k*tau                                                         #df/dc
        df[i,2*N+3] = c*tau                                                         #df/dk
        i=2*N+5
        df[i,2*N+2] = 1
        if Option == 0:
            df[i,N+1]=1                                                             #df/dB0
        elif Option == 1:
            df[i,2*N+4]=-1                                                          #df/dQ
        return df
    
    def down_hill(self, H1):
        # a down-hill modification to get better convergence of the Newton-Ranpson iteration
        # •	Where does it come from: search 'down-hill Newton iteration' via Bing
        # return cc    - the iteration coefficient
        #        Numit - upper limit of iteration steps 
        #                these values are determined based on experience
         if H1 <= 0.3:
             cc = 0.2
             Numit = 200
         elif H1 > 0.3 and H1<= 0.6:
             cc = 0.3
             Numit = 100
         elif H1>0.6 and H1<=0.7:
             cc = 0.1
             Numit = 1000
         elif H1>0.7:
             cc = 0.01
             Numit = 10000
         return cc,Numit  
    
    def sovle_Uw(self, x, y, yc, d, Option):
        # a function to solve the maximum wave-induced velocity
        # •	Where does it come from: Rienecker & Fenton (1981, pages 126)
        # return Uw*math.sqrt(g*d) - maximum wave-induceed velcoity [m/s]
        if Option ==1:
            yc = d
        N = 20
        y1 = y/d
        # print(y)
        Dc = yc/d
        eta0 = x[0]
        eta = np.zeros(N)
        B   = np.zeros(N)
        for ii in range (N):
            eta[ii] = x[ii+1]
        B0  = x[N+1]
        for ii in range(N):
            B[ii]   = x[N+2+ii]
        c   = x[2*N+2]
        k  =  x[2*N+3]
        Q  =  x[2*N+4]
        R  =  x[2*N+5]
            
        temp = 0
        for j in range(N):
            temp = temp + (j+1) * B[j] * math.cosh((j+1) * k * y1) / math.cosh((j+1) * k * Dc) * math.cos((j+1) * k * 0)
        Uw = c + B0 + k * temp
        return Uw*math.sqrt(self.g*d)   
    
    # Other functions for similfication 
    # •	Where does it come from: Rienecker & Fenton (1989, third paragraph, page 125) 
    def    SJM1(self, j, m, k, eta, Dc, N):
        ans = math.sinh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.cos((j+1)*(m+1)*math.pi/N)
        return ans
    def    SJM2(self, j, m, k, eta, Dc, N):
        ans = math.sinh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.sin((j+1)*(m+1)*math.pi/N)
        return ans 
    def    CJM1(self, j, m, k, eta, Dc, N):
        ans = math.cosh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.sin((j+1)*(m+1)*math.pi/N)
        return ans
    def    CJM2(self, j, m, k, eta, Dc, N):
        ans = math.cosh((j+1)*k*eta[m])/math.cosh((j+1)*k*Dc)*math.cos((j+1)*(m+1)*math.pi/N)
        return ans
    def    SJM10(self, j, m, k, eta0, Dc, N):
        ans = math.sinh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.cos((j+1)*0*math.pi/N)
        return ans
    def    SJM20(self, j, m, k, eta0, Dc, N):
        ans = math.sinh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.sin((j+1)*0*math.pi/N)
        return ans 
    def    CJM10(self, j, m, k, eta0, Dc, N):
        ans = math.cosh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.sin((j+1)*0*math.pi/N)
        return ans
    def    CJM20(self, j, m, k, eta0, Dc, N):
        ans = math.cosh((j+1)*k*eta0)/math.cosh((j+1)*k*Dc)*math.cos((j+1)*0*math.pi/N)
        return ans
    
    def run(self):
        T = self.Tp
        H = self.Hs
        d = self.Depth
        Uc = self.Uc
        yc = self.z_ref
        y = self.y
        blimit = self.blimit
        Option = self.Option
        Solution, r_error = self.Fenton(T, H, d, Uc, yc, y, blimit, Option)  # call function 'Fenton' to solve the non-linear equations
        Umax = self.sovle_Uw(Solution,y,yc,d,Option)
        return Umax, r_error

if __name__ == "__main__":
    Tmax = 9
    Hmax = 20
    Depth = 30
    Uc = 0
    # z_ref = 1
    D = 0.0705*2
    Option = 0

    import matplotlib.pyplot as plt

    def blimit(Tmax, Depth, g=9.81):

        # C205 method
        varpi = (4 * np.pi ** 2 * Depth) / (g * Tmax ** 2)
        fvarpi = 1 + 0.666 * varpi + 0.445 * varpi ** 2 - 0.105 * varpi ** 3 + 0.272 * varpi ** 4
        L = Tmax * (g * Depth) ** 0.5 * (fvarpi / (1 + varpi * fvarpi)) ** 0.5
        
        blimit = (((0.141063 * (L / Depth)) + (0.0095721 * (L / Depth) ** 2) + (0.007789 * (L / Depth) ** 3)) / (1 + (0.078834 * (L / Depth)) + (0.0317567 * (L / Depth) ** 2) + (0.0093407 * (L / Depth) ** 3)))
        return L, blimit
    
    # lambda0 = 9.81 * Tmax ** 2 / (2 * np.pi)
    Xi = 0 # / (np.sqrt(blimit * Depth / lambda0))
    
    L, b = blimit(Tmax, Depth)
    print(f'Blimit: {b:.4f} ({b * Depth:.2f}m)')
    print('Wavelength:', L)

    exit()

    for H in np.arange(14, 26):
        print(H)

        zs = np.arange(2, 40, 2)

        us = np.empty_like(zs, dtype=np.float64)
        
        for i, z in enumerate(zs):
            D, z = z, 1
            Hmax = H
            output = Fenton(Tmax, Hmax, Depth, Uc, z, D, Option, b).run()
            us[i] = output[0]
            # print(z, us[i])

        plt.figure()
        plt.scatter(zs / 2, us)
        plt.savefig(f'figs/{H}.png')
        





