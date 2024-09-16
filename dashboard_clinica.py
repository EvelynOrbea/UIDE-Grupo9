import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pathlib
import plotly.graph_objects as go
import numpy as np


############# Funciones de procesamiento y gráficos #############

def preprocesamiento(df):
    df["Admit Source"] = df["Admit Source"].fillna("Not Identified")
    df["Check-In Time"] = pd.to_datetime(df["Check-In Time"], format="%Y-%m-%d %I:%M:%S %p")
    df["Days of Wk"] = df["Check-In Time"].dt.strftime("%A")  # Día de la semana
    df["Check-In Hour"] = df["Check-In Time"].dt.strftime("%H:00")  # Formato de 24 horas (HH:00)
    return df


def crear_heatmap(df, clinic, date_range, admit_source):
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[
        (df['Clinic Name'] == clinic) &
        (df['Check-In Time'].between(start_date, end_date)) &
        (df['Admit Source'] == admit_source)
        ]

    heatmap_data = filtered_df.pivot_table(
        index='Days of Wk', columns='Check-In Hour', aggfunc='size', fill_value=0
    )

    colorscale = [
        [0, '#007bff'],
        [0.2, '#28a745'],
        [0.4, '#ffc107'],
        [0.6, '#17a2b8'],
        [0.8, '#dc3545'],
        [1, '#6c757d']
    ]

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=colorscale,
            hoverongaps=False,
            showscale=False
        )
    )

    annotations = []
    for i, row in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            annotations.append(
                dict(
                    x=col,
                    y=row,
                    text=str(heatmap_data.loc[row, col]),
                    showarrow=False,
                    font=dict(color="black"),
                    xanchor="center",
                    yanchor="middle"
                )
            )

    heatmap_fig.update_layout(
        title='Volumen de pacientes',
        xaxis_nticks=24,
        yaxis=dict(autorange='reversed'),
        xaxis_title="Hora del día (24 horas)",
        yaxis_title="Día de la semana",
        annotations=annotations
    )

    return heatmap_fig


def crear_boxplot(df, clinic, date_range, admit_source):
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[
        (df['Clinic Name'] == clinic) &
        (df['Check-In Time'].between(start_date, end_date)) &
        (df['Admit Source'] == admit_source)
        ]

    boxplot_fig = go.Figure()
    for dept in filtered_df['Department'].unique():
        boxplot_fig.add_trace(go.Box(
            y=filtered_df[filtered_df['Department'] == dept]['Wait Time Min'],
            name=dept
        ))

    boxplot_fig.update_layout(
        title='Distribución de Tiempo de Espera por Departamento',
        yaxis_title='Tiempo de Espera (min)',
        xaxis_title='Departamento'
    )

    return boxplot_fig


def crear_avg_wait_time_barplot(df, wait_time_range):
    filtered_df = df[
        (df['Wait Time Min'] >= wait_time_range[0]) &
        (df['Wait Time Min'] <= wait_time_range[1])
        ]

    avg_wait_time = filtered_df.groupby(['Department', 'Admit Source'])['Wait Time Min'].mean().reset_index()

    bar_fig = go.Figure()
    for dept in avg_wait_time['Department'].unique():
        df_dept = avg_wait_time[avg_wait_time['Department'] == dept]
        bar_fig.add_trace(go.Bar(
            x=df_dept['Admit Source'],
            y=df_dept['Wait Time Min'],
            name=dept
        ))

    bar_fig.update_layout(
        title='Tiempo de Espera Promedio por Departamento y Fuente de Admisión',
        xaxis_title='Fuente de Admisión',
        yaxis_title='Tiempo de Espera Promedio (min)',
        barmode='group'
    )
    return bar_fig


def crear_patient_scatter_plot(df, clinic, date_range):
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[
        (df['Clinic Name'] == clinic) &
        (df['Check-In Time'].between(start_date, end_date))
        ]

    filtered_df['Month'] = filtered_df['Check-In Time'].dt.to_period('M').astype(str)
    filtered_df['Hour'] = filtered_df['Check-In Time'].dt.hour

    monthly_counts = filtered_df.groupby('Month').size().reset_index(name='Patient Count')
    max_month = monthly_counts.loc[monthly_counts['Patient Count'].idxmax()]['Month']
    min_month = monthly_counts.loc[monthly_counts['Patient Count'].idxmin()]['Month']

    monthly_counts_filtered = filtered_df[filtered_df['Month'].isin([max_month, min_month])]
    monthly_counts_filtered = monthly_counts_filtered.groupby(['Month', 'Hour']).size().reset_index(
        name='Patient Count')

    scatter_fig = go.Figure()

    for month in [max_month, min_month]:
        data = monthly_counts_filtered[monthly_counts_filtered['Month'] == month]
        scatter_fig.add_trace(go.Scatter(
            x=data['Hour'],
            y=data['Patient Count'],
            mode='markers+text',
            text=data['Patient Count'],
            textposition='top center',
            name=f'Month: {month}'
        ))

        x = data['Hour']
        y = data['Patient Count']
        coeffs = np.polyfit(x, y, 1)
        trendline = np.polyval(coeffs, x)

        scatter_fig.add_trace(go.Scatter(
            x=x,
            y=trendline,
            mode='lines',
            name=f'Trendline for {month}',
            line=dict(dash='dash')
        ))

    scatter_fig.update_layout(
        title='Número de Pacientes en los Meses con Mayor y Menor Número de Pacientes',
        xaxis_title='Hora del Día',
        yaxis_title='Número de Pacientes',
        legend_title='Mes',
    )

    return scatter_fig


def cargar_datos():
    BASE_PATH = pathlib.Path(__file__).parent.resolve()
    DATA_PATH = BASE_PATH.joinpath("data").resolve()

    df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv.gz"))
    return preprocesamiento(df)


############# Configuración de la Aplicación Dash #############

df = cargar_datos()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            label="Gráficos",
            children=[
                dbc.DropdownMenuItem(dcc.Link("Volumen de Pacientes", href="/volumen", className="nav-link text-dark")),
                dbc.DropdownMenuItem(
                    dcc.Link("Distribución de Tiempo", href="/distribucion", className="nav-link text-dark")),
                dbc.DropdownMenuItem(
                    dcc.Link("Promedio de Tiempo de Espera", href="/avg-wait-time", className="nav-link text-dark")),
                dbc.DropdownMenuItem(dcc.Link("Número de Pacientes por Día y Hora", href="/patient-plot",
                                              className="nav-link text-dark")),
            ],
            nav=True,
            in_navbar=True,
            className="text-dark"
        ),
        dbc.DropdownMenu(
            label="Data",
            children=[
                dbc.DropdownMenuItem(dcc.Link("Tabla de Datos", href="/data", className="nav-link text-dark")),
            ],
            nav=True,
            in_navbar=True,
            className="text-dark"
        ),
    ],
    brand="Clinical Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)


def render_filters():
    return dbc.Card(
        dbc.CardBody([
            html.H5("Filtros", className="card-title"),
            html.Label("Clínica:"),
            dcc.Dropdown(
                id='clinic-dropdown',
                options=[{'label': clinic, 'value': clinic} for clinic in df['Clinic Name'].unique()],
                value=df['Clinic Name'].unique()[0],
                className="mb-3"
            ),
            html.Label("Rango de Fecha:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=df['Check-In Time'].min().date(),
                end_date=df['Check-In Time'].max().date(),
                display_format='YYYY-MM-DD',
                className="mb-3"
            ),
            html.Label("Fuente de Admisión:"),
            dcc.Dropdown(
                id='admit-source-dropdown',
                options=[{'label': source, 'value': source} for source in df['Admit Source'].unique()],
                value=df['Admit Source'].unique()[0],
                className="mb-3"
            )
        ])
    )


def render_filters_avg_wait_time():
    return dbc.Card(
        dbc.CardBody([
            html.H5("Filtros: Tiempo de Espera", className="card-title"),
            html.Label("Rango de Tiempo de Espera (min):"),
            dcc.RangeSlider(
                id='wait-time-range-slider',
                min=0,
                max=df['Wait Time Min'].max(),
                step=1,
                marks={i: str(i) for i in range(0, int(df['Wait Time Min'].max()) + 1, 10)},
                value=[0, df['Wait Time Min'].max()],
                className="mb-3",
                vertical=True,
                verticalHeight=400
            ),
        ])
    )


def render_data_table():
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Filtros: Tabla de Datos", className="card-title"),
                    html.Label("Clínica:"),
                    dcc.Dropdown(
                        id='clinic-dropdown-table',
                        options=[{'label': clinic, 'value': clinic} for clinic in df['Clinic Name'].unique()],
                        value=df['Clinic Name'].unique()[0],
                        className="mb-3"
                    ),
                    html.Label("Fecha de entrada:"),
                    dcc.DatePickerRange(
                        id='date-picker-range-table',
                        start_date=df['Check-In Time'].min().date(),
                        end_date=df['Check-In Time'].max().date(),
                        display_format='YYYY-MM-DD',
                        className="mb-3"
                    ),
                    html.Label("Fuente de Admisión:"),
                    dcc.Dropdown(
                        id='admit-source-dropdown-table',
                        options=[{'label': source, 'value': source} for source in df['Admit Source'].unique()],
                        value=df['Admit Source'].unique()[0],
                        className="mb-3"
                    )
                ])
            ),
            width=3
        ),
        dbc.Col(
            dash_table.DataTable(
                id='data-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                style_table={'overflowX': 'auto'},
                page_size=10,
                # Aplicando estilos personalizados a las filas y columnas
                style_header={
                    'backgroundColor': '#007bff',  # Fondo azul en el encabezado
                    'color': 'white',  # Texto blanco en el encabezado
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'center',  # Centrar texto en las celdas
                    'padding': '8px',  # Añadir relleno para mayor legibilidad
                },
                style_data={
                    'backgroundColor': '#f9f9f9',  # Color de fondo para las celdas de datos
                    'color': '#333'  # Color de texto
                },
                style_data_conditional=[  # Estilos condicionales por fila y columna
                    {
                        'if': {'row_index': 'odd'},  # Filas impares
                        'backgroundColor': '#e0f7fa'  # Color de fondo alterno
                    },
                    {
                        'if': {'column_id': 'Clinic Name'},  # Estilo específico para columna 'Clinic Name'
                        'fontWeight': 'bold',
                        'color': '#0056b3'
                    },
                    {
                        'if': {'column_id': 'Wait Time Min', 'filter_query': '{Wait Time Min} > 60'},  # Valores mayores de 60 minutos en 'Wait Time Min'
                        'backgroundColor': '#ffcccc',  # Fondo rojo claro para advertencia
                        'color': '#ff0000',  # Texto rojo
                        'fontWeight': 'bold'
                    }
                ]
            ),
            width=9,
            style={"margin-top": "15px"}  # Añadiendo margen superior de 15px
        )
    ], style={"margin-top": "15px"})  # Añadiendo margen superior al contenedor general también


@app.callback(
    Output('heatmap-fig', 'figure'),
    [Input('clinic-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('admit-source-dropdown', 'value')]
)
def update_heatmap(clinic, start_date, end_date, admit_source):
    try:
        return crear_heatmap(df, clinic, [pd.to_datetime(start_date), pd.to_datetime(end_date)], admit_source)
    except Exception as e:
        print(f"Error en callback de heatmap: {e}")
        return go.Figure()


@app.callback(
    Output('boxplot-fig', 'figure'),
    [Input('clinic-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('admit-source-dropdown', 'value')]
)
def update_boxplot(clinic, start_date, end_date, admit_source):
    try:
        return crear_boxplot(df, clinic, [pd.to_datetime(start_date), pd.to_datetime(end_date)], admit_source)
    except Exception as e:
        print(f"Error en callback de boxplot: {e}")
        return go.Figure()


@app.callback(
    Output('avg-wait-time-barplot', 'figure'),
    [Input('wait-time-range-slider', 'value')]
)
def update_avg_wait_time_barplot(wait_time_range):
    try:
        return crear_avg_wait_time_barplot(df, wait_time_range)
    except Exception as e:
        print(f"Error en callback de barplot: {e}")
        return go.Figure()


@app.callback(
    Output('patient-scatter-plot', 'figure'),
    [Input('clinic-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_patient_scatter_plot(clinic, start_date, end_date):
    try:
        return crear_patient_scatter_plot(df, clinic, [pd.to_datetime(start_date), pd.to_datetime(end_date)])
    except Exception as e:
        print(f"Error en callback de scatter plot: {e}")
        return go.Figure()


@app.callback(
    Output('data-table', 'data'),
    [Input('clinic-dropdown-table', 'value'),
     Input('date-picker-range-table', 'start_date'),
     Input('date-picker-range-table', 'end_date'),
     Input('admit-source-dropdown-table', 'value')]
)
def update_data_table(clinic, start_date, end_date, admit_source):
    try:
        filtered_df = df[
            (df['Clinic Name'] == clinic) &
            (df['Check-In Time'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) &
            (df['Admit Source'] == admit_source)
            ]
        return filtered_df.to_dict('records')
    except Exception as e:
        print(f"Error en callback de tabla de datos: {e}")
        return []


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    try:
        if pathname == '/volumen':
            return html.Div([
                dbc.Row([
                    dbc.Col(render_filters(), width=3),  # Filtros en el lado izquierdo
                    dbc.Col(dcc.Graph(id='heatmap-fig', figure=crear_heatmap(df, df['Clinic Name'].unique()[0],
                                                                             [df['Check-In Time'].min().date(),
                                                                              df['Check-In Time'].max().date()],
                                                                             df['Admit Source'].unique()[0])), width=9)
                ]),
                html.Div([
                    html.H5("Descripción:"),
                    html.P("Esta página muestra un heatmap que refleja el volumen de pacientes por día de la semana y hora del día."),
                    html.H6("Filtros disponibles:"),
                    html.Ul([
                        html.Li("Clínica: Selecciona la clínica para filtrar los datos."),
                        html.Li("Rango de Fecha: Filtra los datos según el rango de fechas de 'Check-In'."),
                        html.Li("Fuente de Admisión: Filtra según la fuente de admisión del paciente.")
                    ])
                ], className="mt-4")
            ])
        elif pathname == '/distribucion':
            return html.Div([
                dbc.Row([
                    dbc.Col(render_filters(), width=3),  # Filtros en el lado izquierdo
                    dbc.Col(dcc.Graph(id='boxplot-fig', figure=crear_boxplot(df, df['Clinic Name'].unique()[0],
                                                                             [df['Check-In Time'].min().date(),
                                                                              df['Check-In Time'].max().date()],
                                                                             df['Admit Source'].unique()[0])), width=9)
                ]),
                html.Div([
                    html.H5("Descripción:"),
                    html.P("Esta página muestra la distribución del tiempo de espera de los pacientes en distintos departamentos."),
                    html.H6("Filtros disponibles:"),
                    html.Ul([
                        html.Li("Clínica: Selecciona la clínica para filtrar los datos."),
                        html.Li("Rango de Fecha: Filtra los datos según el rango de fechas de 'Check-In'."),
                        html.Li("Fuente de Admisión: Filtra según la fuente de admisión del paciente.")
                    ])
                ], className="mt-4")
            ])
        elif pathname == '/avg-wait-time':
            return html.Div([
                dbc.Row([
                    dbc.Col(render_filters_avg_wait_time(), width=3),  # Filtros en el lado izquierdo
                    dbc.Col(dcc.Graph(id='avg-wait-time-barplot',
                                      figure=crear_avg_wait_time_barplot(df, [0, df['Wait Time Min'].max()])), width=9)
                ]),
                html.Div([
                    html.H5("Descripción:"),
                    html.P("Esta página muestra un gráfico de barras con el tiempo de espera promedio por departamento y fuente de admisión."),
                    html.H6("Filtros disponibles:"),
                    html.Ul([
                        html.Li("Rango de Tiempo de Espera: Filtra los datos según un rango de tiempo de espera específico.")
                    ])
                ], className="mt-4")
            ])
        elif pathname == '/patient-plot':
            return html.Div([
                dbc.Row([
                    dbc.Col(render_filters(), width=3),  # Filtros en el lado izquierdo
                    dbc.Col(dcc.Graph(id='patient-scatter-plot',
                                      figure=crear_patient_scatter_plot(df, df['Clinic Name'].unique()[0],
                                                                        [df['Check-In Time'].min().date(),
                                                                         df['Check-In Time'].max().date()])), width=9)
                ]),
                html.Div([
                    html.H5("Descripción:"),
                    html.P("Esta página muestra un gráfico de dispersión que compara el número de pacientes en los meses con mayor y menor volumen."),
                    html.H6("Filtros disponibles:"),
                    html.Ul([
                        html.Li("Clínica: Selecciona la clínica para filtrar los datos."),
                        html.Li("Rango de Fecha: Filtra los datos según el rango de fechas de 'Check-In'.")
                    ])
                ], className="mt-4")
            ])
        elif pathname == '/data':
            return html.Div([
                render_data_table(),
                html.Div([
                    html.H5("Descripción:"),
                    html.P("Esta página muestra una tabla de datos filtrados por clínica, fecha de entrada y fuente de admisión."),
                    html.H6("Filtros disponibles:"),
                    html.Ul([
                        html.Li("Clínica: Selecciona la clínica para filtrar los datos."),
                        html.Li("Rango de Fecha: Filtra los datos según el rango de fechas de 'Check-In'."),
                        html.Li("Fuente de Admisión: Filtra según la fuente de admisión del paciente.")
                    ])
                ], className="mt-4")
            ])
        else:
            return html.Div([
                html.H2("Bienvenido al Clinical Analytics Dashboard", className="text-center"),
                html.Div(html.Img(src='/assets/eig_logo.png', style={'width': '200px'}), className="text-center mt-3"),
                html.P("Seleccione una opción en la barra de navegación.", className="text-center mt-4"),
                # Caja de texto scrollable con saltos de línea
                html.Div([
                    html.P("1. ¿Cómo estructurarán el Dashboard?"),
                    html.P("""
                        El dashboard está organizado en varias secciones, cada una asociada a un tipo de gráfico o datos:

                        Navbar (Barra de Navegación):
                        Hay una barra de navegación en la parte superior que permite al usuario acceder a diferentes secciones:

                        Gráficos: Incluye opciones como "Volumen de Pacientes", "Distribución de Tiempo", "Promedio de Tiempo de Espera" y "Número de Pacientes por Día y Hora".
                        Data: Contiene la opción "Tabla de Datos" que muestra la tabla con los datos clínicos filtrados.

                        Filtros a la izquierda y contenido a la derecha:
                        Para cada sección de gráficos, el diseño es consistente: los filtros para los gráficos están ubicados en la parte izquierda (en un dbc.Col de ancho 3), y el gráfico correspondiente está en la parte derecha (en un dbc.Col de ancho 9). Esto permite que los usuarios interactúen con los filtros de manera intuitiva y vean los resultados reflejados a la derecha.

                        Página principal (Bienvenida):
                        La página principal muestra un mensaje de bienvenida, un logotipo, y una indicación para seleccionar una opción en la barra de navegación.

                        Esta estructura ofrece una navegación clara, con una separación entre los filtros y las visualizaciones.
                    """),
                    html.P("2. ¿Qué componentes y estilos piensan usar y por qué?"),
                    html.P("""
                        Componentes utilizados:

                        Navbar (Barra de Navegación):
                        Utiliza el componente dbc.NavbarSimple de Dash Bootstrap Components para crear una barra de navegación con menús desplegables, lo que facilita la selección de diferentes gráficos o la tabla de datos. Bootstrap proporciona un estilo limpio y profesional para la interfaz.

                        Filtros (Tarjetas con Dropdowns y Sliders):
                        Los filtros, como el selector de clínicas, la fuente de admisión y el rango de fechas, están dentro de dbc.Card, que es una tarjeta estilizada que agrupa filtros específicos. Esto ayuda a que la interfaz de usuario sea más ordenada y fácil de entender para los usuarios.

                        Dropdowns y DatePickerRange:
                        Los dcc.Dropdown y dcc.DatePickerRange permiten al usuario seleccionar opciones de clínicas, fuente de admisión, y rango de fechas de una manera intuitiva, mientras que el dcc.RangeSlider permite elegir el rango de tiempo de espera.

                        Gráficos interactivos (Graph y DataTable):
                        Los gráficos (dcc.Graph) son interactivos y permiten a los usuarios observar los datos de manera dinámica. Los gráficos incluyen:

                        Un heatmap para mostrar el volumen de pacientes.
                        Un boxplot para visualizar la distribución de tiempos de espera.
                        Un barplot para el tiempo de espera promedio.
                        Un scatter plot para mostrar la cantidad de pacientes por hora en los meses de mayor y menor afluencia.

                        Además, hay una tabla de datos (dash_table.DataTable) que muestra los datos clínicos filtrados, permitiendo una revisión detallada de la información.

                        Estilos:

                        Se usa el tema Bootstrap para garantizar una interfaz visualmente atractiva y con componentes bien alineados.
                        Las tarjetas (dbc.Card) ayudan a organizar los filtros y gráficos en secciones bien definidas.
                        El uso de colores para los gráficos y los estilos predeterminados de Bootstrap mantienen la consistencia visual, haciendo que el dashboard sea agradable y fácil de usar.
                    """),
                    html.P("3. ¿Qué interacciones deben tener estos componentes?"),
                    html.P("""
                        Interacciones con los filtros:

                        Los filtros modifican los gráficos en tiempo real. Por ejemplo, al seleccionar una clínica, una fuente de admisión, o un rango de fechas, los gráficos correspondientes (heatmap, boxplot, barplot, scatter plot) se actualizan dinámicamente para reflejar la información filtrada.

                        El dcc.DatePickerRange, dcc.Dropdown y dcc.RangeSlider permiten interactuar con los datos en función de la selección de fechas, clínicas y tiempos de espera, lo que da a los usuarios pueden seleccionar el tipo de visualización de los datos.

                        Gráficos interactivos:

                        Los gráficos permiten visualizar datos clínicos de una manera gráfica. Estos deben permitir hacer zoom, desplazarse, y mostrar valores detallados mediante herramientas como el hover (cuando el usuario pasa el cursor sobre los puntos o barras en los gráficos).

                        El scatter plot añade una interacción más avanzada mostrando no solo los puntos de datos, sino también una línea de tendencia para los meses con mayor y menor número de pacientes. Esta tendencia también se actualiza de acuerdo con los filtros.

                        Tabla de Datos:
                        La tabla de datos debe permitir búsquedas y filtrados adicionales según las selecciones en los dropdowns y el rango de fechas. Los datos que se muestran en la tabla deben coincidir con las selecciones de filtros que el usuario haya aplicado.
                    """)
                ], style={
                    'border': '1px solid #ccc',  # Borde para la caja de texto
                    'padding': '10px',  # Espacio interno
                    'maxHeight': '300px',  # Altura máxima para la caja de texto
                    'overflowY': 'scroll',  # Hacer que el contenido sea scrollable
                    'margin-top': '15px',  # Margen superior
                    'backgroundColor': '#f8f9fa'  # Color de fondo suave
                })
            ], className="mt-5")
    except Exception as e:
        print(f"Error en la función display_page: {e}")
        return html.Div([
            html.H2("Error al cargar la página", className="text-center"),
            html.P("Hubo un problema al intentar cargar la página. Por favor, intente de nuevo.",
                   className="text-center mt-4")
        ], className="mt-5")


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)