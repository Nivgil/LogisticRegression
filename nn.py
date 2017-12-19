
import torch
import numpy as np
from torch.autograd import Variable
from data_set import LinearSeparableDataSet
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer
from bokeh.plotting import figure, curdoc,output_file, show

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

D_in, D_out = 3, 1
epohcs = round(1e6)

dataset = LinearSeparableDataSet()
dataset = dataset.get_data()

samples_num = dataset.shape[0]

data = dataset[:, 0:2]
data = np.c_[data, np.zeros(samples_num)]
labels = dataset[:, 2]

data = Variable(torch.from_numpy(data).type(dtype))
labels = Variable(torch.from_numpy(labels).type(dtype), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_out),
    torch.nn.Sigmoid(),
)
loss_fn = torch.nn.MSELoss()

learning_rate = 1e-2
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)
stats = torch.FloatTensor(epohcs,D_in+2)
for t in range(epohcs):

    y_pred = model(data)

    loss = loss_fn(y_pred, labels)
    # Statistics
    stats[t, 0] = loss.data[0]
    for param in model.parameters():
        if param.size() == (1,D_in):
            stats[t, 1:D_in+1] = param.data
        else:
            stats[t, D_in+1] = param.data[0]
    # End - Statistics
    if t % 10000 == 0:
        print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

w = stats[:, 1:D_in+1]
w_norm = torch.norm(w, 2, 1)
# import pdb; pdb.set_trace()
max_norm = w_norm[-1]
w_norm = torch.div(w_norm, max_norm[0])
w_norm = w_norm.numpy()
w_norm = np.reshape(w_norm, epohcs)
t = np.arange(1, epohcs+1)

TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"
p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_label='t', y_axis_label='||w(t)||',
           title="The Norm of w(t)", x_axis_type='log')
p.background_fill_color = "#fafafa"
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

p.line(t, w_norm, line_width=2, line_dash='solid')

output_file('w_norm.html')
show(p)













