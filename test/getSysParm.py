import  sys

import torch


def getSysparm():
    args=sys.argv # 获取所有的输入参数列表
    print("the name of code file name:",args[0]) #第一个文件名，为程序名
    if len(args)>1:
        print("input",len(args)-1,"th param")
        for arg in args[1:]:
            print(arg)
        else:
            print("No arguments")

def myargs(*args):# 数组参数
    print(args)

def myargs2(**kwargs):#字典参数
    print(kwargs)

def test1():
    a=3
    b=2
    # a = \ b

def test2():
    a=torch.arange(3).reshape(1,3)
    a1=torch.arange(3., requires_grad=True)
    print(a1)   #tensor([0., 1., 2.], requires_grad=True)
    a2=torch.relu(a1)
    print(a2)   #tensor([0., 1., 2.], grad_fn=<ReluBackward0>)
    a3=a2.backward(torch.ones_like(a1),retain_graph=True)

    print(a3)   #None
    print(a1.grad)  #tensor([0., 1., 1.])

    a22=torch.relu(a2)
    print(a22)  #tensor([0., 1., 2.], grad_fn=<ReluBackward0>)
    a22.backward(torch.ones_like(a2), retain_graph=True)
    print(a1.grad)  #None

    # b=torch.arange(3).reshape(3,1)
    # c=torch.range(1,2)
    # d=torch.randn(2,12).reshape(-1,4)
    # print(a)
    #
    #
    # print(b)
    # print(c)
    # print(d)
    # print(a+b)

def test3():
    X = [[] for _ in range(10)]
    print(X)
if __name__=="__main__":
    # # getSysparm()
    # myargs([1,2],[3,4])
    # myargs2(a=1,b=2)
    # test1()
    test2()