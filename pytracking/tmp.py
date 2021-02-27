import torch

a=torch.tensor([1,2,3,4]).reshape(1,4)
b=a[:,:2]
c=a[:,2:]

# b=[5,6,7,8]
# c=max(a[0:2], b[0:2])
# e=a[0:2]+a[2:4]
# d=min(a[0:2]+a[2:4], b[0:2]+b[2:4])
# [x-1 for x in d]
print(b)
print(c)