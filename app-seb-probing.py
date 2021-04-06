import os
import glob
import random
import json

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State

import dash_vtk
from dash_vtk.utils import to_mesh_state, preset_as_options

import vtk

DATA_PATH = "data"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

vtk_datasets = {}
def _load_vtp(filepath, fieldname=None, point_arrays=[], cell_arrays=[]):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    # Cache mesh for future lookup
    part_name = filepath.split("/")[-1].replace(".vtp", "")
    vtk_datasets[part_name] = reader.GetOutput()

    return to_mesh_state(reader.GetOutput(), fieldname, point_arrays, cell_arrays)

# -----------------------------------------------------------------------------
# GUI setup
# -----------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# -----------------------------------------------------------------------------
# Populate scene
# -----------------------------------------------------------------------------

# vehicle geometry
vehicle_vtk = []
for filename in glob.glob(os.path.join(DATA_PATH, "vehicle") + "/*.vtp"):
    mesh = _load_vtp(filename, point_arrays=['U', 'p'])
    part_name = filename.split("/")[-1].replace(".vtp", "")
    child = dash_vtk.GeometryRepresentation(
        id=f"{part_name}",
        colorMapPreset="erdc_rainbow_bright",
        colorDataRange=[0, 100],
        mapper={"scalarVisibility": False },
        children=[dash_vtk.Mesh(state=mesh,)],
    )
    vehicle_vtk.append(child)

cone_pointer = dash_vtk.GeometryRepresentation(
    property={"color": [1, 0, 0]},
    children=[dash_vtk.Algorithm(id="pointer", vtkClass="vtkConeSource")])

# -----------------------------------------------------------------------------
# 3D Viz
# -----------------------------------------------------------------------------

vtk_view = dash_vtk.View(id="vtk-view", pickingModes=['hover'], children=vehicle_vtk+[cone_pointer])

# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    width=6,
                    children=[
                        html.H2("Vehicle Geometry with OpenFOAM"),
                    ],
                ),
                dbc.Col(
                    width=6,
                    children=[
                        dcc.Dropdown(
                            id="surfcolor",
                            options=[
                                {"label": "solid", "value": "solid"},
                                {"label": "U", "value": "U"},
                                {"label": "p", "value": "p"},
                            ],
                            value="solid",
                        ),
                    ],
                ),
            ],
            style={"margin-top": "15px"},
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    width=12,
                    children=[
                        html.Div(vtk_view, style={"height": "100%", "width": "100%"})
                    ],
                ),
            ],
            style={"margin-top": "15px", "height": "calc(100vh - 33px - 46px - 30px)"},
        ),
        html.Pre(
            id="tooltip",
            style={"position": "absolute", "bottom": "25px", "left": "25px", "zIndex": 1, "color": "white"}
        ),
    ],
)

# -----------------------------------------------------------------------------
# Handle controls
# -----------------------------------------------------------------------------

COLOR_RANGES = {
    'solid': [0, 1],
    'U': [0, 100],
    'p': [-4464, 1700],
}

@app.callback(
    [Output("vtk-view", "triggerRender")]
    + [Output(item.id, "mapper") for item in vehicle_vtk]
    + [Output(item.id, "colorDataRange") for item in vehicle_vtk],
    [
        Input("surfcolor", "value"),
    ],
)
def update_scene(surfcolor):
    triggered = dash.callback_context.triggered

    # update surface coloring
    if triggered and "surfcolor" in triggered[0]["prop_id"]:
        color_range = COLOR_RANGES[triggered[0]["value"]]
        mapper = {
            'colorByArrayName': triggered[0]["value"],
            'scalarMode': 3,
            'interpolateScalarsBeforeMapping': True,
            'scalarVisibility': True,
        }
        if triggered[0]["value"] == "solid":
            mapper = { 'scalarVisibility': False }

        surf_state = [mapper for item in vehicle_vtk]
        color_ranges = [color_range for item in vehicle_vtk]
    else:
        surf_state = [dash.no_update for item in vehicle_vtk]
        color_ranges = [dash.no_update for item in vehicle_vtk]

    return [random.random()] + surf_state + color_ranges


# -----------------------------------------------------------------------------

SCALE_P = 0.0001
SCALE_U = 0.01

@app.callback(
    [
        Output("tooltip", "children"),
        Output("pointer", "state"),
    ],
    [
        Input("vtk-view", "hoverInfo"),
    ])
def probe_data(info):
    cone_state = { 'resolution': 12 }
    if info:
        if 'representationId' not in info:
            return dash.no_update, dash.no_update
        ds_name = info['representationId']
        mesh = vtk_datasets[ds_name]
        if mesh:
            xyx = info['worldPosition']
            idx = mesh.FindPoint(xyx)
            if idx > -1:
                cone_state['center'] = mesh.GetPoints().GetPoint(idx)
                messages = []
                pd = mesh.GetPointData()
                size = pd.GetNumberOfArrays()
                for i in range(size):
                    array = pd.GetArray(i)
                    name = array.GetName()
                    nb_comp = array.GetNumberOfComponents()
                    value = array.GetValue(idx)
                    value_str = f'{array.GetValue(idx):.2f}'
                    norm_str = ''
                    if nb_comp == 3:
                        value = array.GetTuple3(idx)
                        norm = (value[0] ** 2 + value[1] ** 2 + value[2] ** 2) ** 0.5
                        norm_str = f' norm({norm:.2f})'
                        value_str = ', '.join([f'{v:.2f}' for v in value])

                        cone_state['height'] = SCALE_U * norm
                        cone_state['direction'] = [v / norm for v in value]

                    if name == 'p':
                        cone_state['radius'] = array.GetValue(idx) * SCALE_P

                    messages.append(f'{name}: {value_str} {norm_str}')

        if 'height' in cone_state:
            new_center = [v for v in cone_state['center']]
            for i in range(3):
                new_center[i] -= 0.5 * cone_state['height'] * cone_state['direction'][i]
            cone_state['center'] = new_center

        return ['\n'.join(messages)], cone_state
    return [''], cone_state

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
