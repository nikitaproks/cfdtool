import logging

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


from classes.store import Store
from utils.schemas import Settings

logger = logging.getLogger(__name__)
app = dash.Dash(__name__)


def setup_app(store: Store, settings: Settings):
    app.layout = html.Div(
        [
            dcc.Graph(id="cfd-animation"),
            dcc.Store(
                id="cfd-data-store", data={"step": 0}
            ),  # This will store the current step or state
            dcc.Interval(
                id="simulation-interval", interval=50, max_intervals=0
            ),
            html.Button("Start simulation", id="start-simulation-button"),
            html.Button("Reset simulation", id="reset-simulation-button"),
        ]
    )

    @app.callback(
        [
            Output("cfd-data-store", "data", allow_duplicate=True),
            Output("simulation-interval", "n_intervals"),
            Output("simulation-interval", "max_intervals"),
            Output("start-simulation-button", "children"),
            Output("start-simulation-button", "disabled"),
            Output("reset-simulation-button", "disabled"),
        ],
        [
            Input("start-simulation-button", "n_clicks"),
            Input("reset-simulation-button", "n_clicks"),
        ],
        State("start-simulation-button", "children"),
        prevent_initial_call=True,
    )
    def toggle_simulation(start_n_clicks, reset_n_clicks, btn_label):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        if ctx.triggered[0]["prop_id"] == "reset-simulation-button.n_clicks":
            store.set_simulation_running(False)
            store.set_simulation_reset(True)
            store.empty()
            logger.info("User trigger simulation reset")
            return {"step": 0}, 0, 0, "Start Simulation", False, False
        elif ctx.triggered[0]["prop_id"] == "start-simulation-button.n_clicks":
            if btn_label.lower() == "Start Simulation".lower():
                store.set_simulation_running(True)
                logger.info("User trigger simulation start")
                return (
                    dash.no_update,
                    dash.no_update,
                    -1,
                    "Pause Simulation",
                    dash.no_update,
                    True,
                )
            elif btn_label.lower() == "Pause Simulation".lower():
                store.set_simulation_paused(True)
                logger.info("User trigger simulation pause")
                return (
                    dash.no_update,
                    dash.no_update,
                    0,
                    "Continue Simulation",
                    False,
                    False,
                )
            elif btn_label.lower() == "Continue Simulation".lower():
                store.set_simulation_paused(False)
                logger.info("User trigger simulation continue")
                return (
                    dash.no_update,
                    dash.no_update,
                    -1,
                    "Pause Simulation",
                    False,
                    True,
                )

    @app.callback(
        [
            Output("cfd-animation", "figure", allow_duplicate=True),
            Output("cfd-data-store", "data", allow_duplicate=True),
            Output(
                "simulation-interval", "max_intervals", allow_duplicate=True
            ),
            Output(
                "start-simulation-button", "children", allow_duplicate=True
            ),
            Output(
                "start-simulation-button", "disabled", allow_duplicate=True
            ),
            Output(
                "reset-simulation-button", "disabled", allow_duplicate=True
            ),
        ],
        [
            Input("simulation-interval", "n_intervals"),
            Input("reset-simulation-button", "n_clicks"),
        ],
        State("cfd-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_cfd_image(n_intervals, n_clicks, data):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if ctx.triggered[0]["prop_id"] == "reset-simulation-button.n_clicks":
            return (
                {"data": [], "layout": {}},
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        if not store.is_simulation_running():
            store.empty()
            return (
                dash.no_update,
                0,
                0,
                "Start Simulation",
                True,
                False,
            )

        if not store.empty():
            new_frame = store.get()

            fig = go.Figure(data=[go.Heatmap(z=new_frame)])

            data["step"] += 1

            return (
                fig,
                data,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    return app
