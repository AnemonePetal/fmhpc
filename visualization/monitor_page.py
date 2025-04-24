import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import pandas as pd
import vizro.plotly.express as px
from vizro import Vizro
import vizro.models as vm
import numpy as np
from sklearn import metrics
import plotly.graph_objects as go
from vizro.models.types import capture

@capture("graph")
def px_line_enhance(data_frame, x, y, instance, color,flag, hover_data):
    df = data_frame.copy()
    df = df[df['instance'] == instance]
    if not isinstance(y, list):
        y = [y]
    if y==[None]:
        return None
    if flag == 're' or flag==['re']:
        y = [i+'_re' for i in y]
    elif flag == 'gt' or flag==['gt']:
        y = y
    elif flag == ['gt','re']:
        y = y + [i+'_re' for i in y]
    df_melted = df.melt(id_vars=['timestamp', 'instance'], value_vars=y, var_name='y_name', value_name='y_value')
    df_melted['color_label'] = df_melted['y_name'].apply(lambda x: 'Prediction' if 're' in x else 'Ground Truth')
    fig = px.line(df_melted, x='timestamp', y='y_value', color='color_label')

    fig.update_xaxes(
        rangeslider_visible=True,
    )
    fig.update_layout(
        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    return fig

class Monitorpage:
    def __init__(self, args, data):
        self.instances = list(data.train.instance.unique())
        self.default_instance = self.instances[0]
        self.default_metric = args.features[0]
        self.page_train = vm.Page(
            title="Monitor: train",
            components=[
                vm.Graph(id="monitor_line_train", figure=px_line_enhance(data.train, x="timestamp", y=self.default_metric, instance=self.default_instance, color=None, flag=['gt','re'], hover_data=["label"])),
            ],
            controls=[
                vm.Parameter(
                    targets=["monitor_line_train.y"],
                    selector=vm.Dropdown(
                        options=args.features,
                        value=self.default_metric,
                        multi=False,
                    ),
                ),
                vm.Parameter(
                    targets=["monitor_line_train.instance"],
                    selector=vm.Dropdown(
                        options=self.instances,
                        value=self.default_instance,
                        multi=False,
                    ),
                ),
            ],
        )

        self.page_val = vm.Page(
            title="Monitor: val",
            components=[
                vm.Graph(id="monitor_line_val", figure=px_line_enhance(data.val, x="timestamp", y=self.default_metric, instance=self.default_instance, color=None, flag=['gt','re'], hover_data=["label"])),
            ],
            controls=[
                vm.Parameter(
                    targets=["monitor_line_val.y"],
                    selector=vm.Dropdown(
                        options=args.features,
                        value=self.default_metric,
                        multi=False,
                    ),
                ),
                vm.Parameter(
                    targets=["monitor_line_val.instance"],
                    selector=vm.Dropdown(
                        options=self.instances,
                        value=self.default_instance,
                        multi=False,
                    ),
                ),

            ],
        )

        self.page_test = vm.Page(
            title="Monitor: test",
            components=[
                vm.Graph(id="monitor_line_test", figure=px_line_enhance(data.test, x="timestamp", y=self.default_metric, instance=self.default_instance, color=None, flag=['gt','re'], hover_data=["label"])),
            ],
            controls=[
                vm.Parameter(
                    targets=["monitor_line_test.y"],
                    selector=vm.Dropdown(
                        options=args.features,
                        value=self.default_metric,
                        multi=False,
                    ),
                ),
                vm.Parameter(
                    targets=["monitor_line_test.instance"],
                    selector=vm.Dropdown(
                        options=self.instances,
                        value=self.default_instance,
                        multi=False,
                    ),
                ),
            ],
        )

if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from util.argparser import get_args
    from visualization.data_connector import retrieve_data
    args = get_args()
    args.dataset = 'jlab'
    args.model = 'fm'
    args.load_model_path = 'results/jlab_fm_03-13--13-35-29/best_03-13--13-35-29.pt'
    args.save_subdir = 'fm'
    args.batch = 32
    args.deterministic = True
    args.epoch = 100
    args.device = 'cuda:0'
    data = retrieve_data(args)
    monitorpage = Monitorpage(args, data)
    dashboard = vm.Dashboard(
        title="Anomaly Detection",
        pages=[monitorpage.page_test, monitorpage.page_train, monitorpage.page_val],
        navigation=vm.Navigation(
            nav_selector=vm.NavBar(
                items=[
                    vm.NavLink(
                        label="Monitoring",
                        icon="Monitoring",
                        pages=["Monitor: train", "Monitor: val","Monitor: test"],
                    )
                ]
            )
        ),
        )

    Vizro().build(dashboard).run()