import argparse

import numpy as np

import ben_conv_kernels as ben_kernel
import utils



def convert_from_ben(K):
    return K.transpose(0, 3, 1, 2, 4, 5).squeeze()


def five_layer(X):
    K_0 = ben_kernel.initial_gram(X)
    K_1 = ben_kernel.convolve_zp(K_0, 3)
    K_2 = ben_kernel.rectify_relu(K_1)
    K_3 = ben_kernel.convolve_zp(K_2, 3)
    K_4 = ben_kernel.rectify_relu(K_3)
    K_5 = ben_kernel.pool(K_4, 2, 2)
    K_6 = ben_kernel.convolve_zp(K_5, 3)
    K_7 = ben_kernel.rectify_relu(K_6)
    K_8 = ben_kernel.pool(K_7, 2, 2)
    K_9 = ben_kernel.convolve_zp(K_8, 3)
    K_10 = ben_kernel.rectify_relu(K_9)
    K_f = ben_kernel.pool(K_10, 8, 8)
    K_f = K_f.transpose(0, 3, 1, 2, 4, 5).squeeze()
    return K_f

def two_layer(X):
    K_0 = ben_kernel.initial_gram(X)
    K_1 = ben_kernel.convolve_zp(K_0, 3)
    K_2 = ben_kernel.rectify_relu(K_1)
    K_f = ben_kernel.pool(K_2, 32, 32)
    print(K_f.shape)
    K_f = K_f.transpose(0, 3, 1, 2, 4, 5).squeeze()
    return K_f



def main():
    parser = argparse.ArgumentParser("neural_kernels")
    parser.add_argument("--kernel", default="two_layer", help="two_layer, five_layer")
    parser.add_argument("--N", default=8, type=int)
    parser.add_argument("--dataset", default="cifar-10", type=str)
    args = parser.parse_args()
    N = args.N
    kernel = args.kernel
    assert kernel in {"two_layer", "five_layer"}
    if kernel == "two_layer":
        kernel_fn = two_layer
    elif kernel == "five_layer":
        kernel_fn = five_layer
    else:
        assert False
    print("Downloading dataset...")
    data = utils.load_dataset(args.dataset)
    N = args.N

    X_train = data["X_train"][:N]
    K = kernel_fn(X_train)
    print(K)
    print(f"saved kernel to {args.dataset}_{args.N}_{args.kernel}_kernel.npy")
    np.save(f"{args.dataset}_{args.N}_{args.kernel}_kernel.npy", K)



if __name__ == "__main__":
    main()
