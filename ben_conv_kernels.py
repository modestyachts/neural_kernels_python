import numpy as np
from timeit import default_timer as timer
from numba import jit
@jit
def initial_gram(x):
    NumEx=x.shape[0]
    numrow=x.shape[1]
    numcol=x.shape[2]
    numchan=x.shape[3]
    K = np.zeros((NumEx,numrow,numcol,NumEx,numrow,numcol))
    for i in range(NumEx):
        for j in range(numrow):
            for k in range(numcol):
                for l in range(NumEx):
                    for m in range(numrow):
                        for n in range(numcol):
                            for c in range(numchan):
                                K[i,j,k,l,m,n] += x[i,j,k,c]*x[l,m,n,c]
    return K

@jit
def rectify_relu(K, project_sphere=False):
    K_out = np.zeros(K.shape)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for l in range(K.shape[3]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_entry = K[i,j,k,l,m,n]
                            K_norm_i = K[i,j,k,i,j,k]
                            K_norm_l = K[l,m,n,l,m,n]
                            K_scaled = np.minimum(np.maximum(K_entry/np.sqrt(K_norm_i*K_norm_l),-1),1)
                            K_out[i,j,k,l,m,n] =  ((1/np.pi)*np.sqrt(1-K_scaled**2)
                                                +K_scaled*( 1-(1/np.pi)*np.arccos(K_scaled) ))

                            if (not project_sphere):
                                K_out[i,j,k,l,m,n] *=  np.sqrt(K_norm_i) * np.sqrt(K_norm_l)
    return K_out

@jit
def rectify_exp_shifted(K, gamma=1.0, project_sphere=False):
    K_out = np.zeros(K.shape)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for l in range(K.shape[3]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_entry = K[i,j,k,l,m,n]
                            K_norm_i = K[i,j,k,i,j,k]
                            K_norm_l = K[l,m,n,l,m,n]
                            K_scaled = np.minimum(np.maximum(K_entry/np.sqrt(K_norm_i*K_norm_l),-1),1)
                            K_out[i,j,k,l,m,n] =  np.exp(gamma * (K_scaled - 1))

                            if (not project_sphere):
                                K_out[i,j,k,l,m,n] *=  np.sqrt(K_norm_i) * np.sqrt(K_norm_l)
    return K_out

@jit
def rectify_cos(K,gamma=1.0):
    K_out = np.zeros(K.shape)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for l in range(K.shape[3]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_entry = K[i,j,k,l,m,n]
                            K_norm_i = K[i,j,k,i,j,k]
                            K_norm_l = K[l,m,n,l,m,n]
                            K_out[i,j,k,l,m,n]=np.exp(-gamma*(K_norm_i+K_norm_l-2*K_entry))
    return K_out

@jit
def convolve(K,patch_size,stride):
    K_shapes = K.shape
    NumEx = K.shape[0]
    row_out = np.floor((K.shape[1]-patch_size)/stride).astype(int)+1
    col_out = np.floor((K.shape[2]-patch_size)/stride).astype(int)+1
    K_out = np.zeros((NumEx,row_out,col_out,NumEx,row_out,col_out))
    for i in range(NumEx):
        for j in range(row_out):
            for k in range(col_out):
                for l in range(NumEx):
                    for m in range(row_out):
                        for n in range(col_out):
                            for a in range(patch_size):
                                for b in range(patch_size):
                                    K_out[i,j,k,l,m,n] += (1/patch_size**2)*K[i,stride*j+a,stride*k+b,l,stride*m+a,stride*n+b]
    return K_out

@jit
def convolve_zp(K,patch_size):
    # this runs zero padded convolutions. But note that it doesn't yet enable
    # striding. that will be a feature to add. don't need it for Myrtle. ;)
    NumEx = K.shape[0]
    row_out = K.shape[1]
    col_out = K.shape[2]
    K_out = np.zeros(K.shape)
    for i in range(NumEx):
        for j in range(row_out):
            for k in range(col_out):
                for l in range(NumEx):
                    for m in range(row_out):
                        for n in range(col_out):
                            for a in range(-(patch_size-1)//2,(patch_size+1)//2):
                                for b in range(-(patch_size-1)//2,(patch_size+1)//2):
                                    if j+a>=0 and k+b>=0 and m+a>=0 and n+b>=0 and j+a<row_out and k+b<col_out and m+a<row_out and n+b<col_out:
                                        K_out[i,j,k,l,m,n] += (1/patch_size**2)*K[i,j+a,k+b,l,m+a,n+b]
    return K_out


@jit
def pool(K,pool_size,stride):
    K_shapes = K.shape
    NumEx = K.shape[0]
    row_out = np.floor((K.shape[1]-pool_size)/stride).astype(int)+1
    col_out = np.floor((K.shape[2]-pool_size)/stride).astype(int)+1
    K_out = np.zeros((NumEx,row_out,col_out,NumEx,row_out,col_out))
    for i in range(NumEx):
        for j in range(row_out):
            for k in range(col_out):
                for l in range(NumEx):
                    for m in range(row_out):
                        for n in range(col_out):
                            for a in range(pool_size):
                                for b in range(pool_size):
                                    for c in range(pool_size):
                                        for d in range(pool_size):
                                            K_out[i,j,k,l,m,n] += (1/pool_size**4)*K[i,stride*j+a,stride*k+b,l,stride*m+c,stride*n+d]
    return K_out

@jit
def center(K):
    K_out = np.zeros(K.shape)
    K_sum_r = np.zeros(K.shape[0:3])
    K_sum_c = np.zeros(K.shape[3:])
    K_sum_tot = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for l in range(K.shape[3]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_sum_tot += K[i,j,k,l,m,n]
                            K_sum_r[i,j,k] += K[i,j,k,l,m,n]
                            K_sum_c[l,m,n] += K[i,j,k,l,m,n]
    K_sum_r /= np.product(K.shape[0:3])
    K_sum_c /= np.product(K.shape[3:])
    K_sum_tot /= np.product(K.shape)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            for k in range(K.shape[2]):
                for l in range(K.shape[3]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_out[i,j,k,l,m,n] = K[i,j,k,l,m,n] -  K_sum_r[i, j, k] - K_sum_c[l, m, n] + K_sum_tot
    return K_out

@jit
def group_norm(K):
    D = K.shape[1]*K.shape[2]
    K_out = np.zeros(K.shape)
    K_norms_r = 0
    K_norms_c = 0
    K_mean_tot = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[3]):
            K_norms_r = 0
            K_norms_c = 0
            K_mean_tot = 0
            for k in range(K.shape[1]):
                for l in range(K.shape[2]):
                    K_norms_r+=K[i,k,l,i,k,l]/D
                    K_norms_c+=K[j,k,l,j,k,l]/D
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_mean_tot += K[i,k,l,j,m,n]/D**2
            rescale = 1/np.sqrt(K_norms_r)/np.sqrt(K_norms_c)
            for k in range(K.shape[1]):
                for l in range(K.shape[2]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_out[i,k,l,j,m,n] = (K[i,k,l,j,m,n]+K_mean_tot)*rescale
    return K_out

@jit
def group_norm_new(K):
    D = K.shape[1]*K.shape[2]
    K_out = np.zeros(K.shape)
    K_norms_r = 0
    K_norms_c = 0
    for i in range(K.shape[0]):
        for j in range(K.shape[3]):
            K_norms_r = 0
            K_norms_c = 0
            for k in range(K.shape[1]):
                for l in range(K.shape[2]):
                    K_norms_r+=K[i,k,l,i,k,l]/D
                    K_norms_c+=K[j,k,l,j,k,l]/D
            rescale = 1/np.sqrt(K_norms_r)/np.sqrt(K_norms_c)
            for k in range(K.shape[1]):
                for l in range(K.shape[2]):
                    for m in range(K.shape[4]):
                        for n in range(K.shape[5]):
                            K_out[i,k,l,j,m,n] = K[i,k,l,j,m,n]*rescale
    return K_out

def coates_ng(K_0):
    print('convolve')
    K_1 = convolve(K_0,6,1)
    print(K_1.shape)

    print('rectify')
    K_2 = rectify_relu(K_1)
    print(K_2.shape)

    print('pool')
    K_3 = pool(K_2,15,6)
    print(K_3.shape)

    print('convolve')
    K_f = convolve(K_3,3,1)
    print(K_f.shape)

    return K_f

def vgg_ish(K_0):
    print('convolve')
    K_1 = convolve(K_0,3,1)
    print(K_1.shape)

    print('rectify')
    K_2 = rectify_relu(K_1)

    print('convolve')
    K_3 = convolve(K_2,3,2)
    print(K_3.shape)

    print('rectify')
    K_4 = rectify_relu(K_3)

    print('convolve')
    K_5 = convolve(K_4,3,2)
    print(K_5.shape)

    print('rectify')
    K_6 = rectify_relu(K_5)

    print('convolve')
    K_7 = convolve(K_6,3,2)
    print(K_7.shape)

    print('rectify')
    K_8 = rectify_relu(K_7)

    print('convolve')
    K_9 = convolve(K_8,2,1)
    print(K_9.shape)

    print('rectify')
    K_f = rectify_relu(K_9)
    return K_f

def myrtle(K_0):
    # set this flag to False to try a fusion of the first two convs in the
    # myrtle architecture
    unfused_prep = True

    if unfused_prep:
        print('Layer prep: convolve')
        K_1 = convolve_zp(K_0,3)
        print(K_1.shape)

        print('Layer prep: rectify')
        K_2 = rectify_relu(K_1)

        print('Layer 1: convolve')
        K_3 = convolve_zp(K_2,3)
        print(K_3.shape)

        print('Layer 1: rectify')
        K_4 = rectify_relu(K_3)
    else:
        # a thought, it might be possible to just do this instead, fusing the
        # first two convolutions:
        print('Layer prep + 1 merged: convolve')
        K_1 = convolve_zp(K_0,5)
        print(K_1.shape)

        print('Layer prep + 1 merged: rectify')
        K_4 = rectify_relu(K_1)

    print('Layer 1: pool')
    K_5 = pool(K_4,2,2)
    print(K_5.shape)

    print('Layer 2: convolve')
    K_6 = convolve_zp(K_5,3)
    print(K_6.shape)

    print('Layer 2: rectify')
    K_7 = rectify_relu(K_6)

    print('Layer 2: pool')
    K_8 = pool(K_7,2,2)
    print(K_8.shape)

    print('Layer 3: convolve')
    K_9 = convolve_zp(K_8,3)
    print(K_9.shape)

    print('Layer 3: rectify')
    K_10 = rectify_relu(K_9)

    # in the myrtle architeture, for some reason they concatenate two pools
    # at the end. But two concatenated pools are just one big pool. So I've
    # changed it here to be one big pool.
    print('Layer 3: pool')
    K_f = pool(K_10,8,8)

    return K_f

def myrtle_gn(K_0):
    print('Layer prep: convolve')
    K_1 = convolve_zp(K_0,3)
    print(K_1.shape)

    print('Layer prep: group norm')
    K_2 = rectify_relu(group_norm(K_1))
    print(K_2.shape)

    print('Layer 1: convolve')
    K_3 = convolve_zp(K_2,3)
    print(K_3.shape)

    print('Layer 1: group norm')
    K_4 = rectify_relu(group_norm(K_3))
    print(K_4.shape)

    print('Layer 1: pool')
    K_5 = pool(K_4,2,2)
    print(K_5.shape)

    print('Layer 2: convolve')
    K_6 = convolve_zp(K_5,3)
    print(K_6.shape)

    print('Layer 2: group norm')
    K_7 = rectify_relu(group_norm(K_6))
    print(K_7.shape)

    print('Layer 2: pool')
    K_8 = pool(K_7,2,2)
    print(K_8.shape)

    print('Layer 3: convolve')
    K_9 = convolve_zp(K_8,3)
    print(K_9.shape)

    print('Layer 3: group norm')
    K_10 = rectify_relu(group_norm(K_9))
    print(K_10.shape)

    # in the myrtle architeture, for some reason they concatenate two pools
    # at the end. But two concatenated pools are just one big pool. So I've
    # changed it here to be one big pool.
    print('Layer 3: pool')
    K_f = pool(K_10,8,8)

    return K_f

def all_conv(K_0):
    print("all_conv")
    print('convolve')
    # 30
    K_1 = convolve(K_0,3,1)
    print(K_1.shape)

    print('rectify')
    K_2 = rectify_relu(K_1)

    print('convolve')
    # 28
    K_3 = convolve(K_2,3,1)
    print(K_3.shape)
    print(K_3)

    print('rectify')
    K_4 = rectify_relu(K_3)

    print('convolve (with stride)')
    # 12
    K_5 = convolve(K_4,3,2)
    print(K_5.shape)
    print(K_6)

    print('rectify')
    K_6 = rectify_relu(K_5)

    print('convolve')
    #12
    K_7 = convolve(K_6,3,1)
    print(K_7.shape)

    print('rectify')
    K_8 = rectify_relu(K_7)

    print('convolve')
    #10
    K_9 = convolve(K_8,3,1)
    print(K_9.shape)

    print('rectify')
    K_10 = rectify_relu(K_9)
    print(K_10.shape)

    print('convolve (wtih stride)')
    #4
    K_11 = convolve(K_10,3,2)
    K_11 = rectify_relu(K_11)
    print(K_11.shape)

    print('convolve')
    # 2
    K_12 = convolve(K_11,3,1)
    K_12 = rectify_relu(K_12)
    print(K_12.shape)

    print('convolve')
    # 1
    K_f = convolve(K_12,2,1)
    K_f = rectify_relu(K_f)
    print(K_f.shape)
    return(K_f)

def kernel_test():
    NumEx=4
    numrow=32
    numcol=32
    numchan=3

    np.random.seed(42)
    x = np.random.randn(NumEx,numrow,numcol,numchan)

    print('random data')
    print(x.shape)

    for i in range(2):
        T0 = timer()

        print('initial gram')
        K_0 = initial_gram(x)
        print(K_0.shape)

        K_f=myrtle_gn(K_0)

        Tfinal = timer()
        print("{:.1f} seconds".format(Tfinal-T0))

        print("kernel eigenvalues:")
        v,w=np.linalg.eig(np.reshape(K_f,(NumEx,NumEx)))
        print(v)
        print("kernel\n", np.reshape(K_f,(NumEx,NumEx)))

if __name__ == "__main__":
    kernel_test()
