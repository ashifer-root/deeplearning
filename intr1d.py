import torch as tr 
import numpy as np 
import pandas as pd 
"""
here a.dtype gives datatype of elements stored in input
and a.type() gives type of tensor like inttensor,floattensor etc
"""
a=tr.tensor([1,2,3])
print(a,a.dtype,a.type())
b=tr.tensor([1,2,3],dtype=tr.int16)
print(b,b.dtype,b.type())
a=a.type(tr.int16)
print(a,a.dtype,a.type())
print(a.size())#prints size of tensor
print(a.ndimension())#prints dimension of tensor
#reshaping of tensors 
print(a.view(-1,1))
#in veiw we give number of rows n columns if -1 is given it will take original vals
print(a.size())
print(a.ndimension())
arr=np.array([1,2,3])
#converting a numpy array into a tensor
torch_tensor=tr.from_numpy(arr)
print(torch_tensor)
#back to numpy
arr_b=torch_tensor.numpy()
print(arr_b,arr_b.shape)
#to a list
l=torch_tensor.tolist()
print(l)
#each element of a tensor is also a tensor
for i in range(torch_tensor.size()[0]):
    print(torch_tensor[i],torch_tensor[i].type(),torch_tensor[i].ndimension())
#indexing and slicing of a tensor is same as list
c=tr.tensor([1,2,3,4,5,6,7])
print(c)
#slicing
print(c[2:5])
c[2:5]=tr.tensor([300,400,500])
print(c)

#-----MATH OPERATIONS
u=tr.tensor([1,2,3])
v=tr.tensor([4,5,6])
summ=u+v #this does matrix addition
summ2=u+1 #here broadcasting takes place
print(summ,summ2) 
spdt=2*u #scalar multiplication
print(spdt)
#normal pdt
nrpdt=u*v # here each element is multiplied to corresponding elemnt in the
#           other tensor
print(nrpdt)
#dot product or matrix multiplication
dtpdt=tr.dot(u,v) #if both are single rowed then v transpose is taken for multiplication
print(dtpdt)
#functions like mean,deviation,trigos,activation functions like 
#sigmoid,ReLU are also present in library
x=tr.tensor([0,np.pi/2,np.pi])
xsin=tr.sin(x)
print(xsin)
#tensors with ranges
rng=tr.linspace(-2,2,steps=5)
#linspace(start,end,number of steps)
print(rng)
#pandas series to a tensor
pandas_series=pd.Series([0.1, 2, 0.3, 10.1])
new_tensor=tr.from_numpy(pandas_series.values)
print("The new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())
print(new_tensor[1]) #this returns a tensor
print(new_tensor[1].item()) #this gives a number 
print(type(new_tensor[1]))  #this ia a torch.tensor
print(type(new_tensor[1].item())) #this is a float
ls=[[1,2,3],[4,5,6],[7,8,9]]
d2tsr=tr.tensor(ls)
print(d2tsr,d2tsr.dtype,d2tsr.type())
print(d2tsr.ndimension())
print(d2tsr.shape,d2tsr.size())
print(d2tsr.numel()) #gives number of elements in tensor
#indexing
for i in range(d2tsr.size()[0]):
    for j in range(d2tsr.size()[1]):
        print(d2tsr[i][j])
        print(d2tsr[i][j].item())
        print(d2tsr[i,j])
        print(d2tsr[i,j].item())
#slicing
print(d2tsr[:,0]) #column0
print(d2tsr[0,:]) #row0
x=tr.tensor([[1,2],[3,4]])
y=tr.tensor([[5,6],[7,8]])
print(x+y)#does matrix addition
print(x*y) #multiplies each element
print(2*x) #scalar multiplication
print(tr.mm(x,y)) #matrix multiplication
