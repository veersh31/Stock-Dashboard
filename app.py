import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "Stock Analytics Dashboard"
server = app.server  # For deployment

# Import layouts
from layouts.stock_dashboard import stock_dashboard_layout
from layouts.portfolio import portfolio_optimizer_layout
from layouts.ml_predictions import ml_predictions_layout

# Import and register callbacks
from callbacks import register_callbacks
register_callbacks(app)

# App Layout with Navigation
app.layout = dbc.Container([
    # Navbar with gradient background
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.I(className="fas fa-chart-line me-2", style={"font-size": "1.5rem"})),
                    dbc.Col(dbc.NavbarBrand("FINANCIAL ANALYTICS", className="ms-2 fw-bold")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("STOCK DASHBOARD", href="/", className="px-3")),
                    dbc.NavItem(dbc.NavLink("PORTFOLIO OPTIMIZER", href="/optimizer", className="px-3")),
                    dbc.NavItem(dbc.NavLink("ML PREDICTIONS", href="/ml-predictions", className="px-3")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4",
        style={"background": "linear-gradient(90deg, #1e3c72 0%, #2a5298 100%)"}
    ),

    # Location component for multi-page routing
    dcc.Location(id='url', refresh=False),

    # Page content will be dynamically loaded here
    html.Div(id='page-content')
], fluid=True, className="px-4 py-3")

# Navbar toggle callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [dash.dependencies.State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Routing Callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/optimizer':
        return portfolio_optimizer_layout()
    elif pathname == '/ml-predictions':
        return ml_predictions_layout()
    return stock_dashboard_layout()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)