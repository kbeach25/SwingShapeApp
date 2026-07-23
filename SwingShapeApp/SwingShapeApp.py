import reflex as rx
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

### Root directory, needed to access the data created by the pipeline
ss_data_path = Path(__file__).resolve().parent.parent / "data" / "SwingShapeData.csv"
swing_shape_df = pd.read_csv(ss_data_path)

### Create swing shape plot
# only show lines when building/troubleshooting app, remove for real use
show_lines = True

def swing_shape_plotting(id, side):
    mode = "lines+markers" if show_lines else "markers"

    df = swing_shape_df[swing_shape_df["batter"] == id]

    # switch hitter handling
    if side != "":
        df = df[df["side"] == side]
    
    # before anything is selected
    if df.empty:
        return go.Figure()
    
    # create 3d visual
    fig = go.Figure()

    # home plate figure
    x = [-8.5, 8.5, 8.5, 0.0, -8.5]
    y = [0, 0, 0, 0, 0]
    z = [0, 0, -8.5, -17, -8.5]

    fig.add_trace(
        go.Mesh3d(x = x, 
                  y = y, 
                  z = z,
                  i = [0, 0, 2],
                  j = [1, 2, 3],
                  k = [2, 4, 4],
                  color = "white",
                  opacity = 1.0,
                  flatshading = True,
                  showscale = False,
                  )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [-8.5, 8.5, 8.5, 0.0, -8.5, -8.5],
            y=[0, 0, 0, 0, 0, 0],
            z=[0, 0, -8.5, -17.0, -8.5, 0],
            mode="lines",
            line=dict(color="white", width=8),
            showlegend=False,
        )
    )

    fig.update_layout(
        template = "plotly_dark",
        height = 700,
        margin = dict(l = 0, r = 0, t = 0, b = 0),
        scene = dict(
            xaxis_title = "",
            yaxis_title = "",
            zaxis_title = "",
            aspectmode = "data",
        ),
    )

    return fig

### App state class
class AppState(rx.State):
    # Landing page
    current_tab: str = "Landing"

    # State variables
    # Swing shape module variables
    selected_batter: int = 0
    selected_side: str = ""

    def set_landing(self):
        self.current_tab = "Landing"
    
    def set_model_visual(self):
        self.current_tab = "Model Visuals"

    def set_swingshape(self):
        self.current_tab = "Swing Shape"

    # Functions for swing shape tab
    def set_selected_batter(self, name: str):
        self.selected_batter = int(swing_shape_df.loc[swing_shape_df["name"] == name, "batter"].iloc[0])
        self.selected_side = ""

    def set_selected_side(self, side: str):
        self.selected_side = side

    # Get the distinct player names for the swing shape visualizer
    @rx.var
    def player_names(self) -> list[str]:
        return sorted(swing_shape_df[["name", "batter"]].drop_duplicates()["name"].tolist())
    
    # Switch hitter handling for swing shape module
    @rx.var
    def sides(self) -> list[str]:
        if self.selected_batter == 0:
            return []
        
        sides = (swing_shape_df[swing_shape_df["batter"] == self.selected_batter]["side"].unique().tolist())

        if len(sides) == 1:
            return []
        
        return ["RHB", "LHB"]
    
    # Swing figure
    @rx.var
    def swing_fig(self) -> go.Figure:
        return swing_shape_plotting(self.selected_batter, self.selected_side, )
    
    # Semi temporary: show bat speed just to make sure this works
    @rx.var
    def bat_speed(self) -> str:
        if self.selected_batter == 0:
            return ""
        
        # Only one row for non switc hitters
        rows = swing_shape_df[swing_shape_df["batter"] == self.selected_batter]

        if len(rows) == 1:
            return str(rows.iloc[0]["bat_speed"])
        
        if self.selected_side == "":
            return ""
        
        side = "R" if self.selected_side == "RHB" else "L"

        row = rows[rows["side"] == side].iloc[0]

        return str(row["bat_speed"])

    
### Side bar for tab selection
def sidebar():
    return rx.vstack(rx.heading("Tab Options", size = "5"),
                     rx.divider(),
                     
                     rx.button("Landing", width = "100%", on_click = AppState.set_landing, ),
                     rx.button("Expected Performance", width = "100%", on_click = AppState.set_model_visual, ),
                     rx.button("Swing Shape Visualization", width = "100%", on_click = AppState.set_swingshape, ),
                     
                     width = "220px",
                     height = "100vh",
                     padding = "1em",
                     spacing = "4",
                     border_right = "1px solid lightgray",
                     )

### Landing page
def landing_tab():
    return rx.vstack(rx.heading("Landing"), rx.text("Landing Page"), align_items = "start", )

### Model visual page
def model_visual_tab():
    return rx.vstack(rx.heading("Model Visuals"), rx.text("Model Visuals"), align_items = "start", )

### Tab for seeing MLB swing shapes
def swing_shape_tab():

    return rx.vstack(
        # Swing Shape module title
        rx.heading("Swing Shape Visualizer", size = "8", ), 
        
        # Dropdown for names
        rx.select(items = AppState.player_names,
                  placeholder = "Select a player",
                  on_change = AppState.set_selected_batter,
                  width = "350px", 
                  ),

        # If a switch hitter is chosen, give the option to see which side
        rx.cond(AppState.sides != [],
                rx.select(AppState.sides,
                          placeholder = "Choose a side",
                          value = AppState.selected_side,
                          on_change = AppState.set_selected_side,
                          width = "200px",
                          ),
                          ),

        # Display the batter's metrics
        rx.cond(AppState.bat_speed != "",
                rx.text("Bat Speed: ", AppState.bat_speed,),
                ),

        rx.center(rx.plotly(
            data = AppState.swing_fig,
            width = "95%",
            height = "700px",
        ),
        width = "100%",
        ),

                align_items = "start",
                spacing = "5",
        
    )

### Main content
def content():
    return rx.match(
        AppState.current_tab,
        ("Landing", landing_tab()),
        ("Model Visuals", model_visual_tab()),
        ("Swing Shape", swing_shape_tab()),
        landing_tab(), # default tab
    )

### Whole page
def index():
    return rx.hstack(sidebar(), 
                     rx.box(
                         content(),
                         padding = "2em",
                         width = "100%",
                     ),
                     
                     width = "100%",
                     height = "100vh",
                     spacing = "0",
                     )

app = rx.App()
app.add_page(index, route = "/")