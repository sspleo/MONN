{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import defaultdict\n",
    "import os\n",
    "import pickle\n",
    "import sys\n",
    "import numpy as np\n",
    "from rdkit import Chem\n",
    "from rdkit.Chem import AllChem\n",
    "from rdkit import DataStructs\n",
    "from rdkit.ML.Cluster import Butina\n",
    "from scipy.cluster.hierarchy import fcluster, linkage, single\n",
    "from scipy.spatial.distance import pdist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "import networkx as nx\n",
    "from networkx.algorithms import bipartite\n",
    "import matplotlib.pyplot as plt\n",
    "import pylab"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "B = nx.Graph()\n",
    "## node list of bipartite 0\n",
    "B.add_nodes_from(['A', 'B', 'C', 'D'], bipartite = 0)\n",
    "## node list of bipartite 1\n",
    "B.add_nodes_from([1,2,3], bipartite=1)\n",
    "\n",
    "## add edges between them\n",
    "B.add_edges_from([('A',1),('A',2), ('B',1), ('C',1), ('D',1),('D',2),('D',3)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "G = nx.Graph()\n",
    "G.add_nodes_from(valid_cid_list, bipartite = 0)\n",
    "G.add_nodes_from(valid_pid_list, bipartite = 1)\n",
    "G.add_edges_from(valid_pair_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "G_top = valid_cid_list\n",
    "G_pos = nx.bipartite_layout(G, G_top)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "color_map = []\n",
    "for element in G:\n",
    "    if element in G_top:\n",
    "        color_map.append('red')\n",
    "    else:\n",
    "        color_map.append('green')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_graph(graph,file_name):\n",
    "    #initialze Figure\n",
    "    plt.figure(num=None, figsize=(50, 50), dpi=120)\n",
    "    plt.axis('off')\n",
    "    fig = plt.figure(1)\n",
    "    pos = G_pos\n",
    "    #pos = B_pos\n",
    "    nx.draw_networkx_nodes(graph,pos)\n",
    "    nx.draw_networkx_edges(graph,pos)\n",
    "    nx.draw_networkx_labels(graph,pos)\n",
    "\n",
    "    cut = 1.00\n",
    "    xmax = cut * max(xx for xx, yy in pos.values())\n",
    "    ymax = cut * max(yy for xx, yy in pos.values())\n",
    "    plt.xlim(-1, xmax + 0.5)\n",
    "    plt.ylim(-1, ymax+0.2)\n",
    "\n",
    "    plt.savefig(file_name,bbox_inches=\"tight\")\n",
    "    pylab.close()\n",
    "    del fig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_graph(B, 'my_graph.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_graph(G,\"my_graph.pdf\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 6400x4800 with 0 Axes>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 6400x4800 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(num=None, figsize=(80, 60), dpi=80, facecolor='w', edgecolor='k')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Help on function draw_networkx in module networkx.drawing.nx_pylab:\n",
      "\n",
      "draw_networkx(G, pos=None, arrows=True, with_labels=True, **kwds)\n",
      "    Draw the graph G using Matplotlib.\n",
      "    \n",
      "    Draw the graph with Matplotlib with options for node positions,\n",
      "    labeling, titles, and many other drawing features.\n",
      "    See draw() for simple drawing without labels or axes.\n",
      "    \n",
      "    Parameters\n",
      "    ----------\n",
      "    G : graph\n",
      "       A networkx graph\n",
      "    \n",
      "    pos : dictionary, optional\n",
      "       A dictionary with nodes as keys and positions as values.\n",
      "       If not specified a spring layout positioning will be computed.\n",
      "       See :py:mod:`networkx.drawing.layout` for functions that\n",
      "       compute node positions.\n",
      "    \n",
      "    arrows : bool, optional (default=True)\n",
      "       For directed graphs, if True draw arrowheads.\n",
      "       Note: Arrows will be the same color as edges.\n",
      "    \n",
      "    arrowstyle : str, optional (default='-|>')\n",
      "        For directed graphs, choose the style of the arrowsheads.\n",
      "        See :py:class: `matplotlib.patches.ArrowStyle` for more\n",
      "        options.\n",
      "    \n",
      "    arrowsize : int, optional (default=10)\n",
      "       For directed graphs, choose the size of the arrow head head's length and\n",
      "       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute\n",
      "       `mutation_scale` for more info.\n",
      "    \n",
      "    with_labels :  bool, optional (default=True)\n",
      "       Set to True to draw labels on the nodes.\n",
      "    \n",
      "    ax : Matplotlib Axes object, optional\n",
      "       Draw the graph in the specified Matplotlib axes.\n",
      "    \n",
      "    nodelist : list, optional (default G.nodes())\n",
      "       Draw only specified nodes\n",
      "    \n",
      "    edgelist : list, optional (default=G.edges())\n",
      "       Draw only specified edges\n",
      "    \n",
      "    node_size : scalar or array, optional (default=300)\n",
      "       Size of nodes.  If an array is specified it must be the\n",
      "       same length as nodelist.\n",
      "    \n",
      "    node_color : color string, or array of floats, (default='r')\n",
      "       Node color. Can be a single color format string,\n",
      "       or a  sequence of colors with the same length as nodelist.\n",
      "       If numeric values are specified they will be mapped to\n",
      "       colors using the cmap and vmin,vmax parameters.  See\n",
      "       matplotlib.scatter for more details.\n",
      "    \n",
      "    node_shape :  string, optional (default='o')\n",
      "       The shape of the node.  Specification is as matplotlib.scatter\n",
      "       marker, one of 'so^>v<dph8'.\n",
      "    \n",
      "    alpha : float, optional (default=1.0)\n",
      "       The node and edge transparency\n",
      "    \n",
      "    cmap : Matplotlib colormap, optional (default=None)\n",
      "       Colormap for mapping intensities of nodes\n",
      "    \n",
      "    vmin,vmax : float, optional (default=None)\n",
      "       Minimum and maximum for node colormap scaling\n",
      "    \n",
      "    linewidths : [None | scalar | sequence]\n",
      "       Line width of symbol border (default =1.0)\n",
      "    \n",
      "    width : float, optional (default=1.0)\n",
      "       Line width of edges\n",
      "    \n",
      "    edge_color : color string, or array of floats (default='r')\n",
      "       Edge color. Can be a single color format string,\n",
      "       or a sequence of colors with the same length as edgelist.\n",
      "       If numeric values are specified they will be mapped to\n",
      "       colors using the edge_cmap and edge_vmin,edge_vmax parameters.\n",
      "    \n",
      "    edge_cmap : Matplotlib colormap, optional (default=None)\n",
      "       Colormap for mapping intensities of edges\n",
      "    \n",
      "    edge_vmin,edge_vmax : floats, optional (default=None)\n",
      "       Minimum and maximum for edge colormap scaling\n",
      "    \n",
      "    style : string, optional (default='solid')\n",
      "       Edge line style (solid|dashed|dotted,dashdot)\n",
      "    \n",
      "    labels : dictionary, optional (default=None)\n",
      "       Node labels in a dictionary keyed by node of text labels\n",
      "    \n",
      "    font_size : int, optional (default=12)\n",
      "       Font size for text labels\n",
      "    \n",
      "    font_color : string, optional (default='k' black)\n",
      "       Font color string\n",
      "    \n",
      "    font_weight : string, optional (default='normal')\n",
      "       Font weight\n",
      "    \n",
      "    font_family : string, optional (default='sans-serif')\n",
      "       Font family\n",
      "    \n",
      "    label : string, optional\n",
      "        Label for graph legend\n",
      "    \n",
      "    Notes\n",
      "    -----\n",
      "    For directed graphs, arrows  are drawn at the head end.  Arrows can be\n",
      "    turned off with keyword arrows=False.\n",
      "    \n",
      "    Examples\n",
      "    --------\n",
      "    >>> G = nx.dodecahedral_graph()\n",
      "    >>> nx.draw(G)\n",
      "    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout\n",
      "    \n",
      "    >>> import matplotlib.pyplot as plt\n",
      "    >>> limits = plt.axis('off')  # turn of axis\n",
      "    \n",
      "    Also see the NetworkX drawing examples at\n",
      "    https://networkx.github.io/documentation/latest/auto_examples/index.html\n",
      "    \n",
      "    See Also\n",
      "    --------\n",
      "    draw()\n",
      "    draw_networkx_nodes()\n",
      "    draw_networkx_edges()\n",
      "    draw_networkx_labels()\n",
      "    draw_networkx_edge_labels()\n",
      "\n"
     ]
    }
   ],
   "source": [
    "help(nx.draw_networkx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX8AAAD8CAYAAACfF6SlAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi41LCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvSM8oowAAF2xJREFUeJzt3X2MXfV95/H3Z+bOjHEcQsAuYTGOiWqketMopBMEqlKSYiqTSjjadlNQ00JFarUoq2i77MYR2WgXuhFpRJddhe7WJBUPbZZQ1Cbe4oaAS4SUxlmPNywpRIDjJsHEjo0xDuAZex6++8c9d3Lm+Ny513PO3Hn4fV7S0T0Pv3N+X9+Z+ZyHe8+xIgIzM0tL30IXYGZmvefwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswTVEv6SNkt6TtI+SdvatPmwpGclPSPpS3X0a2Zmc6Oqd/hK6geeB64GDgB7gOsj4tlcmw3AQ8CvRsQxST8XEYdn2+7q1atj/fr1lWozM0vN3r17X46INZ3aNWro6zJgX0TsB5D0ILAFeDbX5veBuyPiGECn4AdYv349IyMjNZRnZpYOST/spl0dl30uBF7MTR/I5uVdAlwi6ZuSdkvaXLYhSVsljUgaOXLkSA2lmZlZmV594NsANgDvB64H7pF0TrFRRGyPiOGIGF6zpuNZi5mZzVEd4f8ScFFuem02L+8AsCMixiPin2l+RrChhr7NzGwO6gj/PcAGSRdLGgSuA3YU2nyF5lE/klbTvAy0v4a+zcxsDiqHf0RMAB8DHgW+BzwUEc9Iuk3StVmzR4Gjkp4FngD+fUQcrdq3mZnNTeWves6X4eHh8Ld9zMzOjKS9ETHcqd3yvsP3lVfgnHNAOn34q79a6OrMLDERwZ/t+TOGbh9C/1nTw+Btg9z1rbvo5cH48jzy/+534V3v6q7tr/86/N3fza0fM7MuXfPANXxt/9c6tvvAug+w68ZdSJpTP+ke+d9zT/fBD/DII82zAzOzeTA5NclZf3xWV8EP8MSPnmDoj4eYmJyY17qWV/jv3g1bt575esePw/nn11+PmSXvTZ95E2OTY2e0zvjUOGf9l7PmqaKm5RX+V1wx93UPH4Z//Mf6ajGz5F35xSs5OXlyTutOxASX/o9La67oZ5ZP+N9zT/Vt/PIvV9+GmVnmyQNPVlr/qcNP1VTJ6ZZP+M/lck+ZRfoBuJktLff+33tr2c6d37yzlu0ULZ/wr4ufKWRmNfi9//17tWznlsdvqWU7RQ7/oqO+8djMlr86nue/bAQwBZw1OEhfXx/9/f309/cjaXq6r6+vdGi1mW1wm6XXpjUsZhHB1NQUk5OTy+J1MdRQxyv/EVjEvzoO/xLj4+O1bKcVHK0gae08WuOtodFoMDAwwMDAAI1Gg6GhIQYHBxkcHJwxnm/fWqe1cyoOAH19fTPqAJiammJiYoKpqanp0Gi9Fsdb0/l57X7ZI2LGH2+7obj9sv6LdRVfW/+OVtviOq0bF4vLi0O+Tdl42c8z/1ocL5s+E8U+Z5tud3Nmscb8a7fjZdPtfsfyv9/5aeC0HWjZMFvb/E463ybfNr+sXbvWa+tvpmz7+d/f1t/H5OTk9DAxMcHk5CTj4+NMTU1x6tQpxsfHZwwTExPT85cCh/88yocS1LdTWcxaf4DQPljOJGCKf+z5s7BOwZQPg9n6KmtXDLSy8eI6ZW3L1imbLr53+fnF6XY7mNbvm6TTdmJlO7Y653W7Q51turhDL7YpHji0grl41lB2MJH/O8xvP2UOf6tV6w9soRSDNL+TyF+2a403Go3TzsKKQ+usbHBwcHq8dUbWms4PjUZjuo+BgYHp8fyQ77uoGOCtefllxXZlZwllO4Gy9Yp9tdoU+ypuIyJmHO1OTExMHwWPjo7yxhtv8Nprr/HGG28wOjrK6OgoY2NjnDx5csaR88TExPSRdTHAi7VbfRz+tqy0gmJycnLGq5nN5G/7mJklyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCaol/CVtlvScpH2Sts3S7jckhaThOvo1M7O5qRz+kvqBu4FrgI3A9ZI2lrR7M/Bx4NtV+zQzs2rqOPK/DNgXEfsj4hTwILClpN3twGeBsRr6NDOzCuoI/wuBF3PTB7J50yS9B7goIh6poT8zM6to3j/wldQH/Cnw77pou1XSiKSRI0eOzHdpZmbJqiP8XwIuyk2vzea1vBl4J/ANST8ALgd2lH3oGxHbI2I4IobXrFlTQ2lmZlamjvDfA2yQdLGkQeA6YEdrYUQcj4jVEbE+ItYDu4FrI2Kkhr7NzGwOKod/REwAHwMeBb4HPBQRz0i6TdK1VbdvZmb1q+U/cI+IncDOwrxPt2n7/jr6NDOzufMdvmZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpYgh7+ZWYIc/mZmCXL4m5klyOFvZpagWsJf0mZJz0naJ2lbyfI/kvSspKcl7ZL09jr6NTOzuakc/pL6gbuBa4CNwPWSNhaafQcYjoh3AQ8Df1K1XzMzm7s6jvwvA/ZFxP6IOAU8CGzJN4iIJyLiRDa5G1hbQ79mZjZHdYT/hcCLuekD2bx2bgL+voZ+zcxsjhq97EzSR4Bh4Mo2y7cCWwHWrVvXw8rMzNJSx5H/S8BFuem12bwZJG0CbgWujYiTZRuKiO0RMRwRw2vWrKmhNDMzK1NH+O8BNki6WNIgcB2wI99A0qXAn9MM/sM19GlmZhVUDv+ImAA+BjwKfA94KCKekXSbpGuzZp8DVgF/LekpSTvabM7MzHqglmv+EbET2FmY9+nc+KY6+jEzs3r4Dl8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MElRL+EvaLOk5SfskbStZPiTpy9nyb0taX0e/ZmY2N5XDX1I/cDdwDbARuF7SxkKzm4BjEfHzwH8FPlu1XzMzm7s6jvwvA/ZFxP6IOAU8CGwptNkC3JeNPwxcJUk19G1mZnNQR/hfCLyYmz6QzSttExETwHHgvBr6NjOzOVhUH/hK2ippRNLIkSNHFrocM7Nlq47wfwm4KDe9NptX2kZSA3gLcLS4oYjYHhHDETG8Zs2aGkozM7MydYT/HmCDpIslDQLXATsKbXYAN2Tjvwn8Q0REDX2bmdkcNKpuICImJH0MeBToB/4iIp6RdBswEhE7gC8CD0jaB7xCcwdhZmYLpHL4A0TETmBnYd6nc+NjwL+uoy8zM6tuUX3ga2ZmveHwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBlcJf0rmSHpP0Qvb61pI275b0LUnPSHpa0m9V6dPMzKqreuS/DdgVERuAXdl00QngdyPiXwKbgbsknVOxXzMzq6Bq+G8B7svG7wM+VGwQEc9HxAvZ+I+Bw8Caiv2amVkFVcP//Ig4mI0fAs6frbGky4BB4PsV+zUzswoanRpIehx4W8miW/MTERGSYpbtXAA8ANwQEVNt2mwFtgKsW7euU2lmZjZHHcM/Ija1WybpJ5IuiIiDWbgfbtPubOAR4NaI2D1LX9uB7QDDw8NtdyRmZlZN1cs+O4AbsvEbgK8WG0gaBP4WuD8iHq7Yn5mZ1aBq+N8BXC3pBWBTNo2kYUlfyNp8GPgV4EZJT2XDuyv2a2ZmFXS87DObiDgKXFUyfwT4aDb+l8BfVunHzMzq5Tt8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEEOfzOzBDn8zcwS5PA3M0uQw9/MLEGVwl/SuZIek/RC9vrWWdqeLemApM9X6dPMzKqreuS/DdgVERuAXdl0O7cDT1bsz8zMalA1/LcA92Xj9wEfKmsk6ZeA84GvV+zPzMxqUDX8z4+Ig9n4IZoBP4OkPuBO4JZOG5O0VdKIpJEjR45ULM3MzNppdGog6XHgbSWLbs1PRERIipJ2NwM7I+KApFn7iojtwHaA4eHhsm2ZmVkNOoZ/RGxqt0zSTyRdEBEHJV0AHC5pdgXwPkk3A6uAQUmvR8Rsnw+Ymdk86hj+HewAbgDuyF6/WmwQEb/dGpd0IzDs4DczW1hVr/nfAVwt6QVgUzaNpGFJX6hanJmZzY9KR/4RcRS4qmT+CPDRkvn3AvdW6dPMzKrzHb5mZgly+JuZJcjhb2aWIIe/mVmCHP5mZgmq+j1/swXRulu83V3jU1NTvSzHbMlx+NuSFBEzXs3szPiyj5lZgnzkb0taX18ffX199Pf302g0GBgYYHBwcPp1cHCQFStWMDQ0xMqVK1mxYgUrV67krLPOYmBggEajgSQajQZ9fX1MTk4yPj7OqVOnGBsbY2xsjJMnTzI+Ps7Jkyc5deoU4+PjM4apqSkmJiaYmJhgcnJyepiampp+nZqaIiKmL0f5jMUWmsPfape/Ht9uKMpfxmkFZT4s89vL99FapxW+Y2NjM7Y323gd/8b8eNm8Yu19fX2nLSurqazmsktd3onYXDn8rXbzdT2+tUMoahe6ZTuh/HS+XdlOqSx0i+P5Ib+jaoV868xEEv39/dPTrbOV/FlLa7w1nT+bab329/czMDAwYxgaGqLRaEyf6bTOdlrj+bOgoaGh6fUGBwdpNBoz5uX7ajQapw3t5hd3xidOnOC1117jpz/9KUePHuXgwYMcPHiQQ4cOcejQIV5++WWOHTvGsWPHOH78OCdOnGB0dJTx8XEmJiamDwDa/axby/Lvcf79bGmdfeXPxqzJ4W89Uxa++eBtheRsZw7QDNbi8rL2ZX2Wzc+/dppX/Pe028EVdwxl8/I7jZMnT5buSPJnQWXrlY0Xg7H1ns12JlZc3s067d7vdj+zdj+/VatWsWrVqtJtRQSTk5PTl9tal+VaZ3r5S23j4+OnvQfdaLfjj4gZy850u4udw996xt/Q6a1WcALTr3a62UJ9uQV+nsN/DjodAfX39592yp8/7c+f/hcvBeS30zryyL+2+m+nU9viL3JZ+7I2s63Trl0363bTV7tttmvb7fyyZZ3WPZNtnOm2u91G8WyiU19nehRc9jpbm27nF6dbvwvtlrXrvzU/v37ZtmabX6bTz6E13tqRtrs0tVQ4/EuMjo6yYsWKhS7DbF7NZad1puukuK2pqSlef/11Lv/a5SxmDv8cAf1Av4PfEjDb0b3V4GsLXcDsfJOXmVmCHP5Fo6MLXYGZLQMff+/Ha9nO7/zi79SynaLlE/7XX1/PdnzJx8xqcNcH76plO/f/q/tr2U7R8gn/L32p+jbuuKP6NszMMmtXra20/nkrzqupktMtn/AH+OQnq63/iU/UU4eZGfCDf/uDSusfuuVQPYWUWF7h/5nPwKZNc1t3CX9f18wWp/6+fo5/4vic1j1yyxEa/fP3hczlFf4Ajz0Gt9/effuVKx38ZjZvzl5xNqc+dYpzh87tqv1bBt/C2K1jrH7T6nmta/mFP8CnPtUM9C9+sX2b974XJifhjTd6V5eZJWmgf4Cj247y2rbXuPodV5e2ufLtV/Lqf3iVVz/5KkONoXmvSYv19uTh4eEYGRlZ6DLMzJYUSXsjYrhTu+V55G9mZrOqFP6SzpX0mKQXste3tmm3TtLXJX1P0rOS1lfp18zMqql65L8N2BURG4Bd2XSZ+4HPRcQvAJcBhyv2a2ZmFVQN/y3Afdn4fcCHig0kbQQaEfEYQES8HhEnKvZrZmYVVA3/8yPiYDZ+CDi/pM0lwKuS/kbSdyR9TlJ/STszM+uRjncQSHoceFvJolvzExERksq+OtQA3gdcCvwI+DJwI3Da9zAlbQW2Aqxbt65TaWZmNkcdwz8i2t4yK+knki6IiIOSLqD8Wv4B4KmI2J+t8xXgckrCPyK2A9uh+VXP7v4JZmZ2pqreO7wDuAG4I3v9akmbPcA5ktZExBHgV4GOX+Dfu3fvy5J+WKG21cDLFdbvpaVS61KpE1zrfFgqdULatb69m0aVbvKSdB7wELAO+CHw4Yh4RdIw8AcR8dGs3dXAnTT/s6y9wNaIODXnjrurbaSbGx0Wg6VS61KpE1zrfFgqdYJr7UalI/+IOApcVTJ/BPhobvox4F1V+jIzs/r4Dl8zswQt5/DfvtAFnIGlUutSqRNc63xYKnWCa+1o0T7YzczM5s9yPvI3M7M2lk34d/uQuazt2ZIOSPp8L2vM9d+xVknvlvQtSc9IelrSb/Wwvs2SnpO0T9Jpz2uSNCTpy9nyby/kg/q6qPWPsocJPi1pl6SuvgbX6zpz7X5DUmTfmFsQ3dQq6cPZ+/qMpBr+A+256eLnv07SE9nTBZ6W9MEFqvMvJB2W9E9tlkvSf8/+HU9Les+8FxURy2IA/gTYlo1vAz47S9v/BnwJ+PxirZXmYzE2ZOP/AjgInNOD2vqB7wPvAAaB/wdsLLS5Gfif2fh1wJcX6H3sptYPACuz8T9ciFq7qTNr92bgSWA3MLyI39MNwHeAt2bTP7eIa90O/GE2vhH4wQLV+ivAe4B/arP8g8Df0/w6/OXAt+e7pmVz5E8XD5kDkPRLNJ9B9PUe1VWmY60R8XxEvJCN/5jm3dNrelDbZcC+iNgfzXsxHszqzcvX/zBwlST1oLaijrVGxBPxswcJ7gbW9rhG6O49Bbgd+Cww1sviCrqp9feBuyPiGEBELNRTerupNYCzs/G3AD/uYX0/KyLiSeCVWZpsAe6Ppt00b4y9YD5rWk7h3/Ehc5L6aN5sdksvCyvRzQPxpkm6jOaRzffnuzDgQuDF3PSBbF5pm4iYAI4D5/WgtqJuas27iebRVa91rDM7zb8oIh7pZWElunlPLwEukfRNSbslbe5ZdTN1U+t/Aj4i6QCwE/g3vSntjJ3p73Jl8/dfw8+DGh4ydzOwMyIOzPeBag21trZzAfAAcENETNVbZTokfQQYBq5c6FqKsoOSP6X5wMOloEHz0s/7aZ5JPSnpFyPi1QWtqtz1wL0RcaekK4AHJL3Tf0tLLPyj+kPmrgDeJ+lmYBUwKOn1iGj7AdwC1oqks4FHgFuzU8FeeAm4KDe9NptX1uaApAbN0+mjvSmvtI6WslqRtInmTvfKiDjZo9ryOtX5ZuCdwDeyg5K3ATskXRvNu+V7qZv39ADNa9LjwD9Lep7mzmBPb0qc1k2tNwGbASLiW5JW0HyWzmL7D6W6+l2u03K67NN6yBy0echcRPx2RKyLiPU0L/3cPx/B34WOtUoaBP6WZo0P97C2PcAGSRdnNVxHs968fP2/CfxDZJ9a9VjHWiVdCvw5cO0CXpuetc6IOB4RqyNiffa7uZtmvb0O/o61Zr5C86gfSatpXgba38siM93U+iOyR9BI+gVgBXCkp1V2Zwfwu9m3fi4HjucuDc+Phfjkez4GmtecdwEvAI8D52bzh4EvlLS/kYX7tk/HWoGPAOPAU7nh3T2q74PA8zQ/Y7g1m3cbzUCC5h/QXwP7gP8DvGMBf+6dan0c+EnuPdyxGOsstP0GC/Rtny7fU9G8TPUs8F3gukVc60bgmzS/CfQU8GsLVOf/ovmNvXGaZ043AX9A8wGYrff07uzf8d1e/Px9h6+ZWYKW02UfMzPrksPfzCxBDn8zswQ5/M3MEuTwNzNLkMPfzCxBDn8zswQ5/M3MEvT/AWhF3RmTxVzHAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "nx.draw_networkx(G, pos = G_pos, node_color = color_map, with_labels=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check if B is bipartite set\n",
    "bipartite.is_bipartite(B)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "set0 = bipartite.sets(B)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "top = nx.bipartite.sets(B)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "B_pos = nx.bipartite_layout(B,top)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "color_map = []\n",
    "for node in B:\n",
    "    if node in set0:\n",
    "        color_map.append('red')\n",
    "    else:\n",
    "        color_map.append('green')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "nx.draw_networkx(B, pos = pos, node_color = color_map, with_labels=True, font_weight = 'bold')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']\n",
    "aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']\n",
    "atom_fdim = len(elem_list) + 6 + 6 + 6 + 1\n",
    "bond_fdim = 6\n",
    "max_nb = 6\n",
    "\n",
    "\n",
    "def onek_encoding_unk(x, allowable_set):\n",
    "    if x not in allowable_set:\n",
    "        x = allowable_set[-1]\n",
    "    return list(map(lambda s: x == s, allowable_set))\n",
    "\n",
    "\n",
    "def atom_features(atom):\n",
    "    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) \n",
    "            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) \n",
    "            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])\n",
    "            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])\n",
    "            + [atom.GetIsAromatic()], dtype=np.float32)\n",
    "\n",
    "\n",
    "def bond_features(bond):\n",
    "    bt = bond.GetBondType()\n",
    "    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \\\n",
    "    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)\n",
    "\n",
    "\n",
    "def Mol2Graph(mol):\n",
    "    # convert molecule to GNN input\n",
    "    idxfunc=lambda x:x.GetIdx()\n",
    "\n",
    "    n_atoms = mol.GetNumAtoms()\n",
    "    assert mol.GetNumBonds() >= 0\n",
    "\n",
    "    n_bonds = max(mol.GetNumBonds(), 1)\n",
    "    fatoms = np.zeros((n_atoms,), dtype=np.int32) #atom feature ID\n",
    "    fbonds = np.zeros((n_bonds,), dtype=np.int32) #bond feature ID\n",
    "    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)\n",
    "    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)\n",
    "    num_nbs = np.zeros((n_atoms,), dtype=np.int32)\n",
    "    num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=np.int32)\n",
    "\n",
    "    for atom in mol.GetAtoms():\n",
    "        idx = idxfunc(atom)\n",
    "        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())] \n",
    "\n",
    "    for bond in mol.GetBonds():\n",
    "        a1 = idxfunc(bond.GetBeginAtom())\n",
    "        a2 = idxfunc(bond.GetEndAtom())\n",
    "        idx = bond.GetIdx()\n",
    "        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())] \n",
    "        try:\n",
    "            atom_nb[a1,num_nbs[a1]] = a2\n",
    "            atom_nb[a2,num_nbs[a2]] = a1\n",
    "        except:\n",
    "            return [], [], [], [], []\n",
    "        bond_nb[a1,num_nbs[a1]] = idx\n",
    "        bond_nb[a2,num_nbs[a2]] = idx\n",
    "        num_nbs[a1] += 1\n",
    "        num_nbs[a2] += 1\n",
    "        \n",
    "    for i in range(len(num_nbs)):\n",
    "        num_nbs_mat[i,:num_nbs[i]] = 1\n",
    "\n",
    "    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat\n",
    "\n",
    "\n",
    "def Batch_Mol2Graph(mol_list):\n",
    "    res = list(map(lambda x:Mol2Graph(x), mol_list))\n",
    "    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)\n",
    "    return fatom_list, fbond_list, gatom_list, gbond_list, nb_list\n",
    "\n",
    "\n",
    "def Protein2Sequence(sequence, ngram=1):\n",
    "    # convert sequence to CNN input\n",
    "    sequence = sequence.upper()\n",
    "    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]\n",
    "    output = []\n",
    "    for word in word_list:\n",
    "        if word not in aa_list:\n",
    "            output.append(word_dict['X'])\n",
    "        else:\n",
    "            output.append(word_dict[word])\n",
    "    if ngram == 3:\n",
    "        output = [-1]+output+[-1] # pad\n",
    "    return np.array(output, np.int32)\n",
    "\n",
    "\n",
    "def Batch_Protein2Sequence(sequence_list, ngram=3):\n",
    "    res = list(map(lambda x:Protein2Sequence(x,ngram), sequence_list))\n",
    "    return res\n",
    "\n",
    "\n",
    "def get_mol_dict():\n",
    "    if os.path.exists('../data/mol_dict'):\n",
    "        with open('../data/mol_dict') as f:\n",
    "            mol_dict = pickle.load(f)\n",
    "    else:\n",
    "        mol_dict = {}\n",
    "        mols = Chem.SDMolSupplier('../data/Components-pub.sdf')\n",
    "        for m in mols:\n",
    "            if m is None:\n",
    "                continue\n",
    "            name = m.GetProp(\"_Name\")\n",
    "            mol_dict[name] = m\n",
    "        with open('../data/mol_dict', 'wb') as f:\n",
    "            pickle.dump(mol_dict, f)\n",
    "    #print('mol_dict',len(mol_dict))\n",
    "    return mol_dict\n",
    "\n",
    "\n",
    "def get_pairwise_label(pdbid, interaction_dict):\n",
    "    if pdbid in interaction_dict:\n",
    "        sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])\n",
    "        atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)\n",
    "        atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)\n",
    "        atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)\n",
    "        nonH_position = np.where(atom_element != ('H'))[0]\n",
    "        assert sum(atom_element[nonH_position] != sdf_element) == 0\n",
    "        \n",
    "        atom_name_list = atom_name_list[nonH_position].tolist()\n",
    "        pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)\n",
    "        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:\n",
    "            atom_idx = atom_name_list.index(str(atom_name))\n",
    "            assert atom_idx < len(nonH_position)\n",
    "            \n",
    "            seq_idx_list = []\n",
    "            for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:\n",
    "                if bond_type == bond_type_seq:\n",
    "                    seq_idx_list.append(seq_idx)\n",
    "                    pairwise_mat[atom_idx, seq_idx] = 1\n",
    "        if len(np.where(pairwise_mat != 0)[0]) != 0:\n",
    "            pairwise_mask = True\n",
    "            return True, pairwise_mat\n",
    "    return False, np.zeros((1,1))\n",
    "\n",
    "\n",
    "def get_fps(mol_list):\n",
    "    fps = []\n",
    "    for mol in mol_list:\n",
    "        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)\n",
    "        fps.append(fp)\n",
    "    #print('fingerprint list',len(fps))\n",
    "    return fps\n",
    "\n",
    "\n",
    "def calculate_sims(fps1,fps2,simtype='tanimoto'):\n",
    "    sim_mat = np.zeros((len(fps1),len(fps2))) #,dtype=np.float32)\n",
    "    for i in range(len(fps1)):\n",
    "        fp_i = fps1[i]\n",
    "        if simtype == 'tanimoto':\n",
    "            sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)\n",
    "        elif simtype == 'dice':\n",
    "            sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)\n",
    "        sim_mat[i,:] = sims\n",
    "    return sim_mat\n",
    "\n",
    "\n",
    "def compound_clustering(ligand_list, mol_list):\n",
    "    print 'start compound clustering...'\n",
    "    fps = get_fps(mol_list)\n",
    "    sim_mat = calculate_sims(fps, fps)\n",
    "    #np.save('../preprocessing/'+MEASURE+'_compound_sim_mat.npy', sim_mat)\n",
    "    print 'compound sim mat', sim_mat.shape\n",
    "    C_dist = pdist(fps, 'jaccard')\n",
    "    C_link = single(C_dist)\n",
    "    for thre in [0.3, 0.4, 0.5, 0.6]:\n",
    "        C_clusters = fcluster(C_link, thre, 'distance')\n",
    "        len_list = []\n",
    "        for i in range(1,max(C_clusters)+1):\n",
    "            len_list.append(C_clusters.tolist().count(i))\n",
    "        print 'thre', thre, 'total num of compounds', len(ligand_list), 'num of clusters', max(C_clusters), 'max length', max(len_list)\n",
    "        C_cluster_dict = {ligand_list[i]:C_clusters[i] for i in range(len(ligand_list))}\n",
    "        with open('../preprocessing/'+MEASURE+'_compound_cluster_dict_'+str(thre),'wb') as f:\n",
    "            pickle.dump(C_cluster_dict, f, protocol=0)\n",
    "\n",
    "\n",
    "def protein_clustering(protein_list, idx_list):\n",
    "    print 'start protein clustering...'\n",
    "    protein_sim_mat = np.load('../data/pdbbind_protein_sim_mat.npy').astype(np.float32)\n",
    "    sim_mat = protein_sim_mat[idx_list, :]\n",
    "    sim_mat = sim_mat[:, idx_list]\n",
    "    print 'original protein sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape\n",
    "    #np.save('../preprocessing/'+MEASURE+'_protein_sim_mat.npy', sim_mat)\n",
    "    P_dist = []\n",
    "    for i in range(sim_mat.shape[0]):\n",
    "        P_dist += (1-sim_mat[i,(i+1):]).tolist()\n",
    "    P_dist = np.array(P_dist)\n",
    "    P_link = single(P_dist)\n",
    "    for thre in [0.3, 0.4, 0.5, 0.6]:\n",
    "        P_clusters = fcluster(P_link, thre, 'distance')\n",
    "        len_list = []\n",
    "        for i in range(1,max(P_clusters)+1):\n",
    "            len_list.append(P_clusters.tolist().count(i))\n",
    "        print 'thre', thre, 'total num of proteins', len(protein_list), 'num of clusters', max(P_clusters), 'max length', max(len_list)\n",
    "        P_cluster_dict = {protein_list[i]:P_clusters[i] for i in range(len(protein_list))}\n",
    "        with open('../preprocessing/'+MEASURE+'_protein_cluster_dict_'+str(thre),'wb') as f:\n",
    "            pickle.dump(P_cluster_dict, f, protocol=0)\n",
    "\n",
    "def pickle_dump(dictionary, file_name):\n",
    "    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "MEASURE = 'KIKD' # 'IC50' or 'KIKD'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "20"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mol_dict = get_mol_dict()\n",
    "with open('../data/out7_final_pairwise_interaction_dict','rb') as f:\n",
    "    interaction_dict = pickle.load(f)\n",
    "\n",
    "wlnn_train_list = []\n",
    "atom_dict = defaultdict(lambda: len(atom_dict))\n",
    "bond_dict = defaultdict(lambda: len(bond_dict))\n",
    "word_dict = defaultdict(lambda: len(word_dict))\n",
    "for aa in aa_list:\n",
    "    word_dict[aa]\n",
    "word_dict['X']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 2/5, generating labels...\n",
      "processed sample num 1000\n",
      "processed sample num 2000\n",
      "processed sample num 3000\n",
      "processed sample num 4000\n",
      "processed sample num 5000\n",
      "processed sample num 6000\n",
      "processed sample num 7000\n",
      "processed sample num 8000\n",
      "processed sample num 9000\n",
      "processed sample num 10000\n",
      "processed sample num 11000\n",
      "processed sample num 12000\n",
      "processed sample num 13000\n"
     ]
    }
   ],
   "source": [
    "i = 0\n",
    "pair_info_dict = {}\n",
    "f = open('../data/pdbbind_all_datafile.tsv')\n",
    "print 'Step 2/5, generating labels...'\n",
    "for line in f.readlines():\n",
    "    i += 1\n",
    "    if i % 1000 == 0:\n",
    "        print 'processed sample num', i\n",
    "    pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\\t')\n",
    "    # filter interaction type and invalid molecules\n",
    "    if MEASURE == 'All':\n",
    "        pass\n",
    "    elif MEASURE == 'KIKD':\n",
    "        if measure not in ['Ki', 'Kd']:\n",
    "            continue\n",
    "    elif measure != MEASURE:\n",
    "        continue\n",
    "    if cid not in mol_dict:\n",
    "        print 'ligand not in mol_dict'\n",
    "        continue\n",
    "    mol = mol_dict[cid]\n",
    "    \n",
    "    # get labels\n",
    "    value = float(label)\n",
    "    pairwise_mask, pairwise_mat = get_pairwise_label(pdbid, interaction_dict)\n",
    "    \n",
    "    # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi\n",
    "    if inchi+' '+pid not in pair_info_dict:\n",
    "        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]\n",
    "    else:\n",
    "        if pair_info_dict[inchi+' '+pid][6]:\n",
    "            if pairwise_mask and pair_info_dict[inchi+' '+pid][3] < value:\n",
    "                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]\n",
    "        else:\n",
    "            if pair_info_dict[inchi+' '+pid][3] < value:\n",
    "                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]\n",
    "f.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/leoseo/opt/anaconda3/envs/p2env/lib/python2.7/site-packages/ipykernel_launcher.py:12: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.\n",
      "  if sys.path[0] == '':\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num of neighbor > 6,  84A\n",
      "num of neighbor > 6,  B1R\n",
      "num of neighbor > 6,  670\n",
      "num of neighbor > 6,  B9F\n",
      "num of neighbor > 6,  498\n",
      "num of neighbor > 6,  067\n"
     ]
    }
   ],
   "source": [
    "valid_value_list = []\n",
    "valid_cid_list = []\n",
    "valid_pid_list = []\n",
    "valid_pairwise_mask_list = []\n",
    "valid_pairwise_mat_list = []\n",
    "mol_inputs, seq_inputs = [], []\n",
    "\n",
    "# get inputs\n",
    "for item in pair_info_dict:\n",
    "    pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]\n",
    "    fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)\n",
    "    if fa==[]:\n",
    "        print 'num of neighbor > 6, ', cid\n",
    "        continue\n",
    "    mol_inputs.append([fa, fb, anb, bnb, nbs_mat])\n",
    "    seq_inputs.append(Protein2Sequence(seq,ngram=1))\n",
    "    valid_value_list.append(value)\n",
    "    valid_cid_list.append(cid)\n",
    "    valid_pid_list.append(pid)\n",
    "    valid_pairwise_mask_list.append(pairwise_mask)\n",
    "    valid_pairwise_mat_list.append(pairwise_mat)\n",
    "    wlnn_train_list.append(pdbid)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "valid_pair_list = []\n",
    "for i in range(len(valid_cid_list)):\n",
    "    pair = [valid_cid_list[i], valid_pid_list[i]]\n",
    "    valid_pair_list.append(pair)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6989"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(valid_pair_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6989"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(valid_cid_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "unique_cid_list = list(set(valid_cid_list))\n",
    "unique_pid_list = list(set(valid_pid_list))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "interaction_matrix = np.zeros((len(unique_cid_list), len(unique_pid_list)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5535, 2079)"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "interaction_matrix.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1559"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "unique_cid_list.index(valid_pair_list[0][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(valid_pair_list)):\n",
    "    compound, protein = valid_pair_list[i]\n",
    "    cid_ind = unique_cid_list.index(compound)\n",
    "    pid_ind = unique_pid_list.index(protein)\n",
    "    interaction_matrix[cid_ind, pid_ind] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5535"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(np.sum(interaction_matrix, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "56.0"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max(np.sum(interaction_matrix, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "287.0"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max(np.sum(interaction_matrix, 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [],
   "source": [
    "compound_int_num = np.sum(interaction_matrix, 1)\n",
    "protein_int_num = np.sum(interaction_matrix, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([4407]),)"
      ]
     },
     "execution_count": 253,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where(compound_int_num == max(compound_int_num))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ADP'"
      ]
     },
     "execution_count": 254,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "unique_cid_list[4407]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([   0, 3567, 3566, ..., 1169,  603, 4407])"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.argsort(compound_int_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1.,  1.,  1., ..., 41., 44., 56.])"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compound_int_num[np.argsort(compound_int_num)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 55.,  55.,  61.,  62.,  71.,  71.,  83., 107., 128., 287.])"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "protein_int_num[np.argsort(protein_int_num)][len(protein_int_num)-10:len(protein_int_num)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'index' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-183-216565c10318>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mindex\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprotein_int_num\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margsort\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprotein_int_num\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'index' is not defined"
     ]
    }
   ],
   "source": [
    "protein_int_num[np.argsort(protein_int_num)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "metadata": {},
   "outputs": [],
   "source": [
    "p = Counter(protein_int_num)\n",
    "c = Counter(compound_int_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 244,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "c_list = []\n",
    "p_list = []\n",
    "for i in range(len(c.keys())):\n",
    "    c_list.append([c.keys()[i],c.values()[i]])\n",
    "    \n",
    "p_list = []\n",
    "for i in range(len(p.keys())):\n",
    "    p_list.append([p.keys()[i],p.values()[i]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(c_list)):\n",
    "    c_list[i][0] = int(c_list[i][0])\n",
    "c_list.sort()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[1, 4895],\n",
       " [2, 384],\n",
       " [3, 121],\n",
       " [4, 53],\n",
       " [5, 21],\n",
       " [6, 19],\n",
       " [7, 10],\n",
       " [8, 8],\n",
       " [9, 5],\n",
       " [10, 4],\n",
       " [11, 3],\n",
       " [12, 2],\n",
       " [13, 3],\n",
       " [15, 1],\n",
       " [17, 1],\n",
       " [19, 1],\n",
       " [25, 1],\n",
       " [41, 1],\n",
       " [44, 1],\n",
       " [56, 1]]"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[1, 1156],\n",
       " [2, 359],\n",
       " [3, 163],\n",
       " [4, 75],\n",
       " [5, 60],\n",
       " [6, 60],\n",
       " [7, 29],\n",
       " [8, 32],\n",
       " [9, 19],\n",
       " [10, 17],\n",
       " [11, 19],\n",
       " [12, 11],\n",
       " [13, 7],\n",
       " [14, 4],\n",
       " [15, 9],\n",
       " [16, 2],\n",
       " [17, 3],\n",
       " [18, 7],\n",
       " [19, 3],\n",
       " [20, 2],\n",
       " [21, 1],\n",
       " [22, 3],\n",
       " [23, 8],\n",
       " [25, 1],\n",
       " [26, 1],\n",
       " [28, 2],\n",
       " [29, 1],\n",
       " [30, 2],\n",
       " [31, 2],\n",
       " [33, 2],\n",
       " [34, 1],\n",
       " [35, 1],\n",
       " [36, 1],\n",
       " [37, 1],\n",
       " [43, 3],\n",
       " [44, 1],\n",
       " [47, 1],\n",
       " [55, 2],\n",
       " [61, 1],\n",
       " [62, 1],\n",
       " [71, 2],\n",
       " [83, 1],\n",
       " [107, 1],\n",
       " [128, 1],\n",
       " [287, 1]]"
      ]
     },
     "execution_count": 242,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "p_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "metadata": {},
   "outputs": [],
   "source": [
    "c_list.sort()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(c_list)):\n",
    "    c_list[i][0] = int(c_list[i][0])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(len(c_keys)):\n",
    "    c_keys[i] = int(c_keys[i])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[128.0,\n",
       " 1.0,\n",
       " 2.0,\n",
       " 3.0,\n",
       " 4.0,\n",
       " 5.0,\n",
       " 6.0,\n",
       " 7.0,\n",
       " 8.0,\n",
       " 9.0,\n",
       " 10.0,\n",
       " 11.0,\n",
       " 12.0,\n",
       " 13.0,\n",
       " 14.0,\n",
       " 15.0,\n",
       " 16.0,\n",
       " 17.0,\n",
       " 18.0,\n",
       " 19.0,\n",
       " 20.0,\n",
       " 21.0,\n",
       " 22.0,\n",
       " 23.0,\n",
       " 25.0,\n",
       " 26.0,\n",
       " 28.0,\n",
       " 29.0,\n",
       " 30.0,\n",
       " 31.0,\n",
       " 33.0,\n",
       " 34.0,\n",
       " 35.0,\n",
       " 36.0,\n",
       " 37.0,\n",
       " 43.0,\n",
       " 44.0,\n",
       " 47.0,\n",
       " 55.0,\n",
       " 287.0,\n",
       " 61.0,\n",
       " 62.0,\n",
       " 71.0,\n",
       " 83.0,\n",
       " 107.0]"
      ]
     },
     "execution_count": 233,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "c.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1,\n",
       " 2,\n",
       " 3,\n",
       " 4,\n",
       " 5,\n",
       " 6,\n",
       " 7,\n",
       " 8,\n",
       " 9,\n",
       " 10,\n",
       " 11,\n",
       " 12,\n",
       " 13,\n",
       " 14,\n",
       " 15,\n",
       " 16,\n",
       " 17,\n",
       " 18,\n",
       " 19,\n",
       " 20,\n",
       " 21,\n",
       " 22,\n",
       " 23,\n",
       " 25,\n",
       " 26,\n",
       " 28,\n",
       " 29,\n",
       " 30,\n",
       " 31,\n",
       " 33,\n",
       " 34,\n",
       " 35,\n",
       " 36,\n",
       " 37,\n",
       " 43,\n",
       " 44,\n",
       " 47,\n",
       " 55,\n",
       " 61,\n",
       " 62,\n",
       " 71,\n",
       " 83,\n",
       " 107,\n",
       " 128,\n",
       " 287]"
      ]
     },
     "execution_count": 225,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Annotation',\n",
       " 'Arrow',\n",
       " 'Artist',\n",
       " 'AutoLocator',\n",
       " 'Axes',\n",
       " 'Button',\n",
       " 'Circle',\n",
       " 'Figure',\n",
       " 'FigureCanvasBase',\n",
       " 'FixedFormatter',\n",
       " 'FixedLocator',\n",
       " 'FormatStrFormatter',\n",
       " 'Formatter',\n",
       " 'FuncFormatter',\n",
       " 'GridSpec',\n",
       " 'IndexLocator',\n",
       " 'Line2D',\n",
       " 'LinearLocator',\n",
       " 'Locator',\n",
       " 'LogFormatter',\n",
       " 'LogFormatterExponent',\n",
       " 'LogFormatterMathtext',\n",
       " 'LogLocator',\n",
       " 'MaxNLocator',\n",
       " 'MultipleLocator',\n",
       " 'Normalize',\n",
       " 'NullFormatter',\n",
       " 'NullLocator',\n",
       " 'PolarAxes',\n",
       " 'Polygon',\n",
       " 'Rectangle',\n",
       " 'ScalarFormatter',\n",
       " 'Slider',\n",
       " 'Subplot',\n",
       " 'SubplotTool',\n",
       " 'Text',\n",
       " 'TickHelper',\n",
       " 'Widget',\n",
       " '_INSTALL_FIG_OBSERVER',\n",
       " '_IP_REGISTERED',\n",
       " '__builtins__',\n",
       " '__doc__',\n",
       " '__file__',\n",
       " '__name__',\n",
       " '__package__',\n",
       " '_auto_draw_if_interactive',\n",
       " '_autogen_docstring',\n",
       " '_backend_mod',\n",
       " '_backend_selection',\n",
       " '_hold_msg',\n",
       " '_imread',\n",
       " '_imsave',\n",
       " '_interactive_bk',\n",
       " '_pylab_helpers',\n",
       " '_setp',\n",
       " '_setup_pyplot_info_docstrings',\n",
       " '_show',\n",
       " '_string_to_bool',\n",
       " 'absolute_import',\n",
       " 'acorr',\n",
       " 'angle_spectrum',\n",
       " 'annotate',\n",
       " 'arrow',\n",
       " 'autoscale',\n",
       " 'autumn',\n",
       " 'axes',\n",
       " 'axhline',\n",
       " 'axhspan',\n",
       " 'axis',\n",
       " 'axvline',\n",
       " 'axvspan',\n",
       " 'bar',\n",
       " 'barbs',\n",
       " 'barh',\n",
       " 'bone',\n",
       " 'box',\n",
       " 'boxplot',\n",
       " 'broken_barh',\n",
       " 'cla',\n",
       " 'clabel',\n",
       " 'clf',\n",
       " 'clim',\n",
       " 'close',\n",
       " 'cm',\n",
       " 'cohere',\n",
       " 'colorbar',\n",
       " 'colormaps',\n",
       " 'colors',\n",
       " 'connect',\n",
       " 'contour',\n",
       " 'contourf',\n",
       " 'cool',\n",
       " 'copper',\n",
       " 'csd',\n",
       " 'cycler',\n",
       " 'dedent',\n",
       " 'delaxes',\n",
       " 'deprecated',\n",
       " 'disconnect',\n",
       " 'division',\n",
       " 'docstring',\n",
       " 'draw',\n",
       " 'draw_all',\n",
       " 'draw_if_interactive',\n",
       " 'errorbar',\n",
       " 'eventplot',\n",
       " 'figaspect',\n",
       " 'figimage',\n",
       " 'figlegend',\n",
       " 'fignum_exists',\n",
       " 'figtext',\n",
       " 'figure',\n",
       " 'fill',\n",
       " 'fill_between',\n",
       " 'fill_betweenx',\n",
       " 'findobj',\n",
       " 'flag',\n",
       " 'gca',\n",
       " 'gcf',\n",
       " 'gci',\n",
       " 'get',\n",
       " 'get_backend',\n",
       " 'get_cmap',\n",
       " 'get_current_fig_manager',\n",
       " 'get_figlabels',\n",
       " 'get_fignums',\n",
       " 'get_plot_commands',\n",
       " 'get_scale_docs',\n",
       " 'get_scale_names',\n",
       " 'getp',\n",
       " 'ginput',\n",
       " 'gray',\n",
       " 'grid',\n",
       " 'hexbin',\n",
       " 'hist',\n",
       " 'hist2d',\n",
       " 'hlines',\n",
       " 'hold',\n",
       " 'hot',\n",
       " 'hsv',\n",
       " 'imread',\n",
       " 'imsave',\n",
       " 'imshow',\n",
       " 'inferno',\n",
       " 'install_repl_displayhook',\n",
       " 'interactive',\n",
       " 'ioff',\n",
       " 'ion',\n",
       " 'is_numlike',\n",
       " 'ishold',\n",
       " 'isinteractive',\n",
       " 'jet',\n",
       " 'legend',\n",
       " 'locator_params',\n",
       " 'loglog',\n",
       " 'magma',\n",
       " 'magnitude_spectrum',\n",
       " 'margins',\n",
       " 'matplotlib',\n",
       " 'matshow',\n",
       " 'minorticks_off',\n",
       " 'minorticks_on',\n",
       " 'mlab',\n",
       " 'new_figure_manager',\n",
       " 'nipy_spectral',\n",
       " 'np',\n",
       " 'over',\n",
       " 'pause',\n",
       " 'pcolor',\n",
       " 'pcolormesh',\n",
       " 'phase_spectrum',\n",
       " 'pie',\n",
       " 'pink',\n",
       " 'plasma',\n",
       " 'plot',\n",
       " 'plot_date',\n",
       " 'plotfile',\n",
       " 'plotting',\n",
       " 'polar',\n",
       " 'print_function',\n",
       " 'prism',\n",
       " 'psd',\n",
       " 'pylab_setup',\n",
       " 'quiver',\n",
       " 'quiverkey',\n",
       " 'rc',\n",
       " 'rcParams',\n",
       " 'rcParamsDefault',\n",
       " 'rc_context',\n",
       " 'rcdefaults',\n",
       " 'register_cmap',\n",
       " 'rgrids',\n",
       " 'savefig',\n",
       " 'sca',\n",
       " 'scatter',\n",
       " 'sci',\n",
       " 'semilogx',\n",
       " 'semilogy',\n",
       " 'set_cmap',\n",
       " 'setp',\n",
       " 'show',\n",
       " 'silent_list',\n",
       " 'six',\n",
       " 'specgram',\n",
       " 'spectral',\n",
       " 'spring',\n",
       " 'spy',\n",
       " 'stackplot',\n",
       " 'stem',\n",
       " 'step',\n",
       " 'streamplot',\n",
       " 'style',\n",
       " 'subplot',\n",
       " 'subplot2grid',\n",
       " 'subplot_tool',\n",
       " 'subplots',\n",
       " 'subplots_adjust',\n",
       " 'summer',\n",
       " 'suptitle',\n",
       " 'switch_backend',\n",
       " 'sys',\n",
       " 'table',\n",
       " 'text',\n",
       " 'thetagrids',\n",
       " 'tick_params',\n",
       " 'ticklabel_format',\n",
       " 'tight_layout',\n",
       " 'time',\n",
       " 'title',\n",
       " 'tricontour',\n",
       " 'tricontourf',\n",
       " 'tripcolor',\n",
       " 'triplot',\n",
       " 'twinx',\n",
       " 'twiny',\n",
       " 'unicode_literals',\n",
       " 'uninstall_repl_displayhook',\n",
       " 'violinplot',\n",
       " 'viridis',\n",
       " 'vlines',\n",
       " 'waitforbuttonpress',\n",
       " 'warn_deprecated',\n",
       " 'warnings',\n",
       " 'winter',\n",
       " 'xcorr',\n",
       " 'xkcd',\n",
       " 'xlabel',\n",
       " 'xlim',\n",
       " 'xscale',\n",
       " 'xticks',\n",
       " 'ylabel',\n",
       " 'ylim',\n",
       " 'yscale',\n",
       " 'yticks']"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dir(plt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
