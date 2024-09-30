import numpy as np

class LCPSolver():
    def __init__(self,M,q,maxIter = 100):
        n = len(q)
        self.T = np.hstack((np.eye(n),-M,-np.ones((n,1)),q.reshape((n,1))))
        self.n = n
        self.wPos = np.arange(n)
        self.zPos = np.arange(n,2*n)
        self.W = 0
        self.Z = 1
        self.Y = 2
        self.Q = 3
        TbInd = np.vstack((self.W*np.ones(n,dtype=int),
                           np.arange(n,dtype=int)))
        TnbInd = np.vstack((self.Z*np.ones(n,dtype=int),
                            np.arange(n,dtype=int)))
        DriveInd = np.array([[self.Y],[0]])
        QInd = np.array([[self.Q],[0]])
        self.Tind = np.hstack((TbInd,TnbInd,DriveInd,QInd))
        self.maxIter = maxIter
        
    def solve(self):
        initVal = self.initialize()
        if not initVal:
            return np.zeros(self.n),0,'Solution Found'
        
        for k in range(self.maxIter):
            stepVal = self.step()
            if self.Tind[0,-2] == self.Y:
                # Solution Found
                z = self.extractSolution()
                return z,0,'Solution Found'
            elif not stepVal:
                initVal = self.initialize()
                if not initVal:
                    return self.extractSolution(),0,'Secondary ray Solution'
                return None,1,'Secondary ray found'
            
        return None,2,'Max Iterations Exceeded'
        
    def initialize(self):
        q = self.T[:,-1]
        minQ = np.min(q)
        if minQ < 0:
            ind = np.argmin(q)
            self.clearDriverColumn(ind)
            self.pivot(ind)
            
            return True
        else:
            return False
            
    def step(self):
        q = self.T[:,-1]
        a = self.T[:,-2]
        ind = np.nan
        minRatio = np.inf
        for i in range(self.n):
            if a[i] > 0:
                newRatio = q[i] / a[i]
                if newRatio < minRatio:
                    ind = i
                    minRatio = newRatio
                    
        if minRatio < np.inf:
            self.clearDriverColumn(ind)
            self.pivot(ind)
            return True
        else:
            return False
        
    def extractSolution(self):
        z = np.zeros(self.n)
        q = self.T[:,-1]
        for i in range(self.n):
            if self.Tind[0,i] == self.Z:
                z[self.Tind[1,i]] = q[i]
        return z
    def partnerPos(self,pos):
        v,ind = self.Tind[:,pos]
        if v == self.W:
            ppos = self.zPos[ind]
        elif v == self.Z:
            ppos = self.wPos[ind]
        else:
            ppos = None
        return ppos
        
    def pivot(self,pos):
        ppos = self.partnerPos(pos)
        if ppos is not None:
            self.swapColumns(pos,ppos)
            self.swapColumns(pos,-2)
            return True
        else:
            self.swapColumns(pos,-2)
            return False
            
        
    
    def swapMatColumns(self,M,i,j):
        Mi = np.array(M[:,i],copy=True)
        Mj = np.array(M[:,j],copy=True)
        M[:,i] = Mj
        M[:,j] = Mi
        return M
    
    def swapPos(self,v,ind,newPos):
        if v == self.W:
            self.wPos[ind] = newPos % (2*self.n+2)
        elif v == self.Z:
            self.zPos[ind] = newPos % (2*self.n+2)
            
    
    def swapColumns(self,i,j):
        iInd = self.Tind[:,i]
        jInd = self.Tind[:,j]
        
        v,ind = iInd
        self.swapPos(v,ind,j)
        v,ind = jInd
        self.swapPos(v,ind,i)
        
        self.Tind = self.swapMatColumns(self.Tind,i,j)
        self.T = self.swapMatColumns(self.T,i,j)
     
    
    def clearDriverColumn(self,ind):
        a = self.T[ind,-2]
        self.T[ind] /= a
        for i in range(self.n):
            if i != ind:
                b = self.T[i,-2]
                self.T[i] -= b * self.T[ind]
                
        
    def ind2str(self,indvec):
        v,pos = indvec
        if v == self.W:
            s = 'w%d' % pos
        elif v == self.Z:
            s = 'z%d' % pos
        elif v == self.Y:
            s = 'y'
        else:
            s = 'q'
        return s
    
    def indexStringArray(self):
        indstr = np.array([self.ind2str(indvec) for indvec in self.Tind.T],dtype=object)
        return indstr
        
    def indexedTableau(self):
        indstr = self.indexStringArray()
        return np.vstack((indstr,self.T))
    def __repr__(self):
        IT = self.indexedTableau()
        return IT.__repr__()
    def __str__(self):
        IT = self.indexedTableau()
        return IT.__str__()
    

if __name__ == '__main__':
    import time

    M=[]
    w=[]

    M.append(np.array([[1,0,0],
                       [2,3,0],
                       [4,5,6]]))
    w.append(np.array([9,8,7]))

    M.append(np.array([[2,1],
                       [0,2]]))
    w.append(np.array([-1,-2]))

    M.append(np.array([[0,1,0],
                       [3,2,0],
                       [4,5,6]]))
    w.append(np.array([1,0,10]))
    M.append(np.array([
        [ 11.05231716,   7.36332368,  -0.0230811 ,   0.0230811 ,   8.10146633 ,       -8.10146633,   0.        ,   0.        ],
        [  7.36332368,  11.05231716,  -8.10146633,   8.10146633,   0.0230811  ,       -0.0230811 ,   0.        ,   0.        ],
        [ -0.0230811 ,  -8.10146633,  13.51018284, -13.51018284,   9.92582368 ,       -9.92582368,   1.        ,   0.        ],
        [  0.0230811 ,   8.10146633, -13.51018284,  13.51018284,  -9.92582368 ,        9.92582368,   1.        ,   0.        ],
        [  8.10146633,   0.0230811 ,   9.92582368,  -9.92582368,  13.51018284 ,      -13.51018284,   0.        ,   1.        ],
        [ -8.10146633,  -0.0230811 ,  -9.92582368,   9.92582368, -13.51018284 ,       13.51018284,   0.        ,   1.        ],
        [  1.        ,   0.        , -1.         , -1.         , 0.        ,    0.     ,      0.      ,     0.     ],
        [  0.        ,   1.        ,  0.         ,  0.         ,-1.        ,-1.        ,   0.         ,  0.        ],
     ]))
    w.append(np.array([0.01173916,  0.01173916,  0.00336367, -0.00336367, -0.00336367,  0.00336367, 0.          ,0.        ]))
     
    M.append(np.array([
    [ 11.00464668  ,10.9791795  , -0.0163959  ,  0.0163959  ,  0.59639008  , -0.59639008   ,0.           ,0.        ],
    [ 10.9791795   ,11.00652213 , -0.56083722 ,  0.56083722 , -0.01910746  ,  0.01910746   ,0.           ,0.        ],
    [ -0.0163959   ,-0.56083722 , 13.55785332 ,-13.55785332 , 13.42777673  ,-13.42777673   ,1.           ,0.        ],
    [  0.0163959   , 0.56083722 ,-13.55785332 , 13.55785332 ,-13.42777673  , 13.42777673   ,1.           ,0.        ],
    [  0.59639008  ,-0.01910746 , 13.42777673 ,-13.42777673 , 13.55597787  ,-13.55597787   ,0.           ,1.        ],
    [ -0.59639008  , 0.01910746 ,-13.42777673 , 13.42777673 ,-13.55597787  , 13.55597787   ,0.           ,1.        ],
    [  1.          , 0.         , -1.         , -1.         ,  0.          ,  0.           ,0.           ,0.        ],
    [  0.          , 1.         ,  0.         ,  0.         , -1.          , -1.           ,0.           ,0.        ],
    ]))
    w.append( np.array([ 2.4779501,   2.00023205,  0.00454444, -0.00454444,  0.00538398, -0.00538398,  0.,          0.,        ]))

    M.append(np.array([
        [ 11.05943026,   9.86001965,  -0.01350688,   0.01350688,   4.74091356,   -4.74091356,   0.,           0.,        ],
        [  9.86001965,  11.05943026,  -4.74091356,   4.74091356,   0.01350688,   -0.01350688,   0.,           0.,        ],
        [ -0.01350688,  -4.74091356,  13.50306974, -13.50306974,  12.42251965,  -12.42251965,   1.,           0.,        ],
        [  0.01350688,   4.74091356, -13.50306974,  13.50306974, -12.42251965,   12.42251965,   1.,           0.,        ],
        [  4.74091356,   0.01350688,  12.42251965, -12.42251965,  13.50306974,  -13.50306974,   0.,           1.,        ],
        [ -4.74091356,  -0.01350688, -12.42251965,  12.42251965, -13.50306974,   13.50306974,   0.,           1.,        ],
        [  1.,           0.,          -1.,          -1.,           0.,            0.,           0.,           0.,        ],
        [  0.,           1.,           0.,           0.,          -1.,           -1.,           0.,           0.,        ],
        ]))

    w.append(np.array([ 0.5317451,   0.52435773,  0.01625221, -0.01625221,  0.01625221, -0.01625221,  0.,          0.,        ]))



    M.append(np.array([
        [ 1.10608701e+01,  9.45026243e+00, -9.96056021e-03,  9.96056021e-03,  5.50973705e+00, -5.50973705e+00,  0.00000000e+00,  0.00000000e+00],
        [ 9.45026243e+00,  1.10545477e+01, -5.52230918e+00,  5.52230918e+00,  2.08273226e-02, -2.08273226e-02,  0.00000000e+00,  0.00000000e+00],
        [-9.96056021e-03, -5.52230918e+00,  1.35016299e+01, -1.35016299e+01,  1.20114849e+01, -1.20114849e+01,  1.00000000e+00,  0.00000000e+00],
        [ 9.96056021e-03,  5.52230918e+00, -1.35016299e+01,  1.35016299e+01, -1.20114849e+01,  1.20114849e+01,  1.00000000e+00,  0.00000000e+00],
        [ 5.50973705e+00,  2.08273226e-02,  1.20114849e+01, -1.20114849e+01,  1.35079523e+01, -1.35079523e+01,  0.00000000e+00,  1.00000000e+00],
        [-5.50973705e+00, -2.08273226e-02, -1.20114849e+01,  1.20114849e+01, -1.35079523e+01,  1.35079523e+01,  0.00000000e+00,  1.00000000e+00],
        [ 1.00000000e+00,  0.00000000e+00, -1.00000000e+00, -1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.00000000e+00, -1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        ]))
    w.append(np.array([ 0.34799771,  0.39083768,  0.01375643, -0.01375643,  0.02151533, -0.02151533,  0.,          0.,        ]))




    for i in range(len(M)):

        start = time.time()
        sol = LCPSolver(M[i],w[i]).solve()

        print('Case #',i+1)
        print("\tTime spent: {:.7f}s".format(time.time() - start))
        print('\t',sol)
