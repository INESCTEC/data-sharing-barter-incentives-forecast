# flake8: noqa
def plot_session_stack_bar(df, x_id, y_id, color_id, title_id):
    import plotly.express as px
    fig = px.bar(df, x=x_id, y=y_id, color=color_id, 
                    title=title_id, height=400)
    return fig

def plot_boxplot(df, x_id, y_id, color_id, title_id):
    import plotly.express as px
    fig = px.box(df, x=x_id, 
                        y=y_id, 
                        color=color_id,
                        title=title_id, height=400)
    return fig

def plot_avg_barline(df, x_string, y1_string, y2_string, y3_string, title_):
    import plotly.graph_objects as go
    x_ = list(df[x_string].values)
    y1 = list(df[y1_string].values)
    y2 = list(df[y2_string].values)
    y3 = list(df[y3_string].values)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_,
                    y=y1,
                    name=y1_string,
                    marker_color='rgb(15, 110, 109)',
                    ))
    fig.add_trace(go.Bar(x=x_,
                    y=y2,
                    name=y2_string,
                    marker_color='rgb(106, 110, 255)',
                    ))
    fig.add_trace(
        go.Scatter(
            x=x_,
            y=y3,
            marker_color='red',
            name=y3_string
        ))
    fig.update_layout(title_text=title_, 
    xaxis_title=x_string, yaxis_title=y3_string
)
    return fig

def plot_performance(df, x_string, y1_string, y2_string, title_):
    import plotly.graph_objects as go
    x_ = list(df[x_string].values)
    y1 = list(df[y1_string].values)
    y2 = list(df[y2_string].values)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_,
                    y=y1,
                    name=y1_string,
                    marker_color='blue', opacity=0.4
                    ))
    fig.add_trace(go.Bar(x=x_,
                    y=y2,
                    name=y2_string,
                    marker_color='red', opacity=0.4
                    ))
    fig.add_hline(y=df[y1_string].mean(), line_width=3, line_dash="dash", line_color="blue")
    fig.add_hline(y=df[y2_string].mean(), line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(title_text=title_, 
    xaxis_title=x_string, yaxis_title= 'elapsed_time'
    )
    return fig