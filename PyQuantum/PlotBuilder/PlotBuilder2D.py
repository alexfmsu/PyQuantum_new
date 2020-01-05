# =====================================================================================================================
# EXAMPLES:

# ---------------------------------------------------------------------------------------------------------------------
'''
from PyQuantum.Tools.PlotBuilder2D import *

data = [{
    'x':[1,2,3], 
    'y':[3,4,5]
}]

plt_builder = PlotBuilder2D({
    "title":'title',
    "x_title":'x_title',
    "y_title":'y_title',
    "as_annotation": False,
    # "to_file":'1.png',
    "html":'1.html',
    "data":data
})

plt_builder.make_plot()
'''
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------

# =====================================================================================================================
# plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
# import chart_studio.plotly.graph_objs as go
# import chart_studio.plotly as py
# import chart_studio

# ---------------------------------------------------------------------------------------------------------------------
# PyQuantum.Tools
from PyQuantum.PlotBuilder.Token import *
# ---------------------------------------------------------------------------------------------------------------------



# -------------------------------
def sup(s):
    if not isinstance(s, str):
        s = str(s)

    return '<sup>' + s + '</sup>'
# -------------------------------


# -------------------------------
def sub(s):
    if not isinstance(s, str):
        s = str(s)

    return '<sub>' + s + '</sub>'
# -------------------------------




plotly.tools.set_credentials_file(token[0]['login'], token[0]['key'])
# chart_studio.tools.set_credentials_file(token[0]['login'], token[0]['key'])


class PlotBuilder2D:
    def __init__(self, args):
        # ------------- TITLE -------------------
        if 'title' in args:
            self.title = args['title']
        else:
            self.title = 'title'
        # ------------- TITLE -------------------
        


        # ------------- X_TITLE -----------------
        if 'x_title' in args:
            self.x_title = args['x_title']
        else:
            self.x_title = 'x_title'
        # ------------- X_TITLE -----------------



        # ------------- Y_TITLE -----------------
        if 'y_title' in args:
            self.y_title = args['y_title']
        else:
            self.y_title = 'y_title'
        # ------------- Y_TITLE -----------------


        
        # ---------------------------- DATA -----------------------------------
        if 'data' not in args:
            print('\'data\' not in args')
            exit(0)

        self.data = args['data']
        
        if isinstance(self.data, dict):
            self.data = [self.data]

        if isinstance(self.data, list):
            if len(self.data) == 0:
                print('len(\'data\') == 0')
                exit(0)

            # if len(self.data) == 1:
            #     # self.data = self.data[0]

            #     self.x_min = min(self.data['x'])
            #     self.y_min = min(self.data['y'])

            #     self.x_max = max(self.data['x'])
            #     self.y_max = max(self.data['x'])
            else:
                for value in self.data:
                    if 'x' not in value.keys():
                        print('\'x\' not in i.keys()')
                        exit(0)

                    if 'y' not in value.keys():
                        print('\'y\' not in i.keys()')
                        exit(0)
                
                self.x_min = min([min(value['x']) for value in self.data])
                self.y_min = min([min(value['y']) for value in self.data])
                
                self.x_max = max([max(value['x']) for value in self.data])
                self.y_max = max([max(value['y']) for value in self.data])
        else:
            print('\'data\' is not list or dict')
            exit(0)
        # ---------------------------- DATA -----------------------------------

        # ------------- ONLINE ------------------
        if 'online' not in args:
            self.online = False
        elif 'online' in [True, False]:
            self.online = args['online']
        # ------------- ONLINE ------------------



        # ------------- TO_FILE -----------------
        if 'to_file' not in args:
            self.to_file = False
        elif 'to_file' in [True, False]:
            self.to_file = args['to_file']
        # ------------- TO_FILE -----------------

        
        
        # ------------- HTML --------------------
        if 'html' not in args:
            self.html = 'tmp.html'
        else:
            self.html = args['html']
        # ------------- HTML --------------------



        # ------------- AS_ANNOTATION --------------------
        if 'as_annotation' in args:
            self.as_annotation = args['as_annotation']
            self.x_title_annotation = self.x_title
            self.y_title_annotation = self.y_title

            self.x_title = None
            self.y_title = None
        else:
            self.as_annotation = None
        # ------------- AS_ANNOTATION --------------------

        # self.y_range = args['y_range'] if 'y_range' in args else [0, 1]

    def make_plot(self):
        layout = dict(
            margin=dict(l=100),
            annotations=[
                {
                    'xref': 'paper',
                    'yref': 'paper',
                    # 'x': -0.1,
                    # 'x': -0.095,
                    'x': -0.1175,
                    'xanchor': 'left',
                    'y': 0.5,
                    'yanchor': 'middle',
                    'text': self.y_title_annotation,
                    'showarrow': False,
                    'font': dict(
                        # --------------------------------
                        family='Lato',
                        # family="Courier New, monospace",
                        # family='Open Sans, sans-serif',
                        # --------------------------------

                        size=20,

                        color="#222"
                    ),
                }, {
                    'xref': 'paper',
                    'yref': 'paper',
                    # 'x': 0,
                    'x': 0.5,
                    'xanchor': 'center',
                    # 'y': -0.3,
                    'y': -0.175,
                    'yanchor': 'bottom',
                    'text': self.x_title_annotation,
                    'showarrow': False,
                    'font': dict(
                        # --------------------------------
                        family='Lato',
                        # family="Courier New, monospace",
                        # family='Open Sans, sans-serif',
                        # --------------------------------

                        size=20,

                        color="#222"
                    ),
                }],
            orientation=0,
            width=1024,
            height=600,
            titlefont=dict(
                # --------------------------------
                family='Lato',
                # family="Courier New, monospace",
                # family='Open Sans, sans-serif',
                # --------------------------------

                size=20,

                color="#222"
            ),
            title='<b>' + self.title + '</b>',
            xaxis={
                'title': self.x_title,
                'linewidth': 2,
                'ticks': 'outside',
                # 'zeroline': True,
                'showline': True,
                'zeroline': False,
                # 'showline': False,
                'titlefont': dict(
                    family='Lato',
                    #     color="#000000",
                    color="#222",
                    size=18,
                ),
                'tickfont': dict(
                    family='Lato',
                    #     color="#000000",
                    color="#222",
                    size=16,
                ),
                'range': [self.x_min, self.x_max],
            },
            yaxis={
                'title': self.y_title,
                # 'tickangle': 0,
                'range': [self.y_min, self.y_max],
                # 'autorange': True,
                'linewidth': 2,
                'ticks': 'outside',

                # 'zeroline': True,
                'showline': True,
                # 'ticks': 'outside',
                'zeroline': False,
                'titlefont': dict(
                    family='Lato',
                    #     color="#000000",
                    color="#222",
                    size=18,
                ),
                'tickfont': dict(
                    family='Lato',
                    #     color="#000000",
                    color="#222",
                    size=16,
                ),
                # 'tickangle': 90,
            },
            legend=go.layout.Legend(
                # x=0,
                # y=1,
                # traceorder="normal",
                font=dict(
                    # family="sans-serif",
                    size=16,
                    color="#222",
                    family='Lato',

                ),
                # bgcolor="LightSteelBlue",
                # bordercolor="Black",
                # borderwidth=2
            ),
        )

        fig = dict(data=self.data, layout=layout)

        # fig['layout'].update_layout(
        #     legend=go.layout.Legend(
        #         # x=0,
        #         # y=1,
        #         # traceorder="normal",
        #         font=dict(
        #             # family="sans-serif",
        #             size=22,
        #             # color="black"
        #         ),
        #         # bgcolor="LightSteelBlue",
        #         # bordercolor="Black",
        #         # borderwidth=2
        #     )
        # )

        if self.online:
            py.plot(fig, filename=self.html)
        # py.plot(fig, filename=filename)
        else:
            if self.to_file:
                done = False

                while not done:
                    try:
                        py.image.save_as(fig, filename=self.to_file)
                        done = True
                    except plotly.exceptions.PlotlyRequestError:
                        change_token()
                        break
            else:
                plotly.offline.plot(fig, filename=self.html)
            # plotly.offline.init_notebook_mode()

# =====================================================================================================================
