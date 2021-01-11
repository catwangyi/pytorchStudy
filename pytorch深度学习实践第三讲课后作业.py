import torch


class MyNet():
    def __init__(self):
        self.w = torch.Tensor([[0.5, 0.4]])
        self.w.requires_grad = True
        self.b = torch.Tensor([[2]])

    def forward(self, x):
        x = torch.Tensor([[x**2], [x]])
        out = self.w.mm(x)
        return out+self.b

    def loss(self, x, y):
        y_pred = self.forward(x)
        return (y_pred - y) ** 2


x_data = [1.0, 2.0, 3.0]#假设y=2*x^2+3*x+1,4的输出为45
y_data = [6.0, 15.0, 28.0]

net = MyNet()
print("before training", 4, net.forward(4).data[0, 0].item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = net.loss(x, y)
        l.backward(torch.ones_like(torch.Tensor([[1.0]])))
        # print('\tgrad',x,y,net.w.grad[0].item())
        net.w.data = net.w.data - 0.01 * net.w.grad.data
        net.w.grad.data.zero_()#数据清零

    print(epoch,"progress loss：", l.data.item())

print("after training", 4, net.forward(4).data[0, 0].item())
