from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from math import floor, ceil

def save_tree(dtree, feature_names, class_names): 
    temp = tempfile.TemporaryFile(suffix='.dot')
    temp.close()
    export_graphviz(
       decision_tree=dtree,
       out_file=temp.name,
       feature_names=feature_names,
       class_names=class_names,
       rounded=True,
       filled=True
    )
    return temp.name

def visualize_tree(dtree, feature_names, class_names): 
    file_name = save_tree(dtree, feature_names, class_names)
    return Source.from_file(file_name)

def plot_results(train_score, test_score, train_label = None, test_label = None, xlabel = None, ylabel = None, xvalues = None, ax = None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    if not xvalues: xvalues = range(1, len(train_score) + 1)
    ax.plot(xvalues,train_score,'-', label = train_label)
    ax.plot(xvalues,test_score,'-', label = test_label)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel or ylabel: ax.legend()

def grid_plot(elements, width, height, make_plot, figsize = (20, 20)):
    if (width * height < len(elements)): raise ValueError("Insufficient room in the grid")
    fig, axes = plt.subplots(height, width, figsize = figsize)
    for index, element in enumerate(elements):
        ax=axes[index // width][index % width]
        make_plot(ax, element, index)
    return axes

def adjust_bounds(xmin, xmax, ymin, ymax):
    xmin, ymin = floor(xmin), floor(ymin)
    xmax, ymax = ceil(xmax + .1*abs(xmax)), ceil(ymax + .1*abs(ymax))
    return xmin, xmax, ymin, ymax

def plot_probabilities(classifier, X, y, cmap, levels = 10):
    xs, ys =  X[:, 0], X[:, 1]
    xmin, xmax, ymin, ymax = adjust_bounds(xs.min(), xs.max(), ys.min(), ys.max())
    xx, yy = np.meshgrid(np.arange(xmin, xmax, .1), np.arange(ymin, ymax, .1))
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap = 'viridis', levels = levels)
    plt.colorbar()
    plt.scatter(xs, ys, c=y, cmap=cmap)

def plot_linear_svc_boundaries(svm_clf, xmin, xmax, ymin = -np.inf, ymax=np.inf, support_vectors = True):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    xmin, xmax, ymin, ymax = adjust_bounds(xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = xmin + .1, xmax - .1, ymin + .1, ymax - .1 # Stop plot from painting outside the lines

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    margin = 1/w[1]

    def decision_boundary(x):
        return -w[0] / w[1] * x - b / w[1]
    
    def gutter_up(x):
        return decision_boundary(x) + margin
    
    def gutter_down(x):
        return decision_boundary(x) - margin
    
    def limit(xs, f):
        return np.array([x for x in xs if f(x) > ymin and f(x) < ymax])

    x_decision = limit(x0, decision_boundary)
    x_up = limit(x0, gutter_up)
    x_down = limit(x0, gutter_down)
    svs = svm_clf.support_vectors_

    plt.plot(x_decision, decision_boundary(x_decision), "k-", linewidth=2)
    plt.plot(x_up, gutter_up(x_up), "k--", linewidth=2)
    plt.plot(x_down, gutter_down(x_down), "k--", linewidth=2)
    if support_vectors: plt.plot(svs[:, 0], svs[:, 1], 'Xr')
