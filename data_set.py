import numpy as np
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer
from bokeh.plotting import figure, curdoc,output_file, show

class LinearSeparableDataSet:
    m = -1
    n = 1
    n_points = 16
    data = 1
    seed = 3233
    def __init__(self,n_points = None,m = None,n = None):
        if m is not None:
            self.m = m
        if n is not None:
            self.n = n
        if n_points is not None:
            self.n_points = n_points


        np.random.seed(self.seed)
        self.data = np.zeros(3 * self.n_points).reshape(self.n_points,3)
        # self.data[:, 0] = np.abs(self.n/self.m) * np.random.randn(self.n_points)
        # self.data[:, 1] = np.abs(self.n) * np.random.randn(self.n_points)
        self.data[:, 0] = np.random.randn(self.n_points)
        self.data[:, 1] = np.random.randn(self.n_points)

        self._classify()

    def _classify(self):

        y = self._f(self.data[:,0])
        self.data[:, 2] = (y < self.data[:,1]).astype(int)

    def _f(self,x):
        return self.m * x + self.n

    def update(attr, old, new):
        return


    def visualize(self):
        TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"
        p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                   toolbar_location="above", x_axis_label='X_1', y_axis_label='X_2',
                   title="Linearly Separable Data")
        p.background_fill_color = "#fafafa"
        p.select(BoxSelectTool).select_every_mousemove = False
        p.select(LassoSelectTool).select_every_mousemove = False

        d_positive = self.data[self.data[:, 2] == 1]
        d_negative = self.data[self.data[:, 2] == 0]
        p.scatter(d_positive[:, 0], d_positive[:, 1], size=6, color="red", alpha=0.6)
        p.scatter(d_negative[:, 0], d_negative[:, 1], size=6, color="blue", alpha=0.6)

        output_file('data.html')
        show(p)

    def get_data(self):
        return self.data.copy()



# def main():
#     seed = 123
#     n_points = 16
#     # import pdb; pdb.set_trace()
#     np.random.seed(seed)
#
#     x = LinearSeparableDataSet(n_points)
#
#     x.visualize()
#     data = x.get_data()
#     print(data)



# if __name__ == '__main__':
#     main()
