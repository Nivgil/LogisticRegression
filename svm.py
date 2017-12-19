from data_set import LinearSeparableDataSet
from sklearn import svm
import numpy as np
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer, Range1d
from bokeh.plotting import figure, curdoc,output_file, show

def svm_classifier(data,labels):
    clf = svm.SVC(kernel='linear',C = np.inf)
    clf.fit(data,labels)
    return clf

def get_svm_figure():
    n_points = 16
    dataset = LinearSeparableDataSet(n_points)
    dataset = dataset.get_data()
    data = dataset[:,0:2]
    labels = dataset[:,2]
    clf = svm_classifier(data,labels)

    # import pdb;pdb.set_trace()
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-3, 3, num = 100)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin


    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"
    p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=10, min_border_left=50,
               toolbar_location="above" , x_axis_label='X_1', y_axis_label='X_2',
               title="Linearly Separable Data", y_range=Range1d(-2.5, 3.5))
    p.background_fill_color = "#fafafa"
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False

    d_positive = dataset[dataset[:, 2] == 1]
    d_negative = dataset[dataset[:, 2] == 0]
    p.scatter(d_positive[:, 0], d_positive[:, 1], size=6, color="red", alpha=0.6)
    p.scatter(d_negative[:, 0], d_negative[:, 1], size=6, color="blue", alpha=0.6)
    p.line(xx, yy, legend="SVM.", line_width=2,line_dash='solid')
    p.line(xx, yy_up, line_width=2,line_dash='dashed')
    p.line(xx, yy_down, line_width=2, line_dash='dashed')

    output_file('data.html')
    return p


if __name__ == '__main__':
    get_svm_figure()


