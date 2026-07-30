import reflex as rx
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib

### Root directory, needed to access the data created by the pipeline
ss_data_path = Path(__file__).resolve().parent.parent / "data" / "SwingShapeData.csv"
swing_shape_df = pd.read_csv(ss_data_path)

### Pitch by pitch data
pbp_data_path = Path(__file__).resolve().parent.parent / "data" / "ModelData.csv"
pbp_df = pd.read_csv(pbp_data_path)

### Swing cluster data and create summary table
cluster_data_path = Path(__file__).resolve().parent.parent / "data" / "clusters.csv"
cluster_df = pd.read_csv(cluster_data_path)

cluster_summary = (
    cluster_df.groupby("GMM_Cluster", as_index = False)
              .agg(
                  bat_speed = ("bat_speed", "mean"),
                  attack_angle = ("attack_angle", "mean"),
                  vba = ("vba", "mean"),
                  ttc = ("ttc", "mean"),
                  swing_length = ("swing_length", "mean"),
                  avg = ("avg", "mean"),
                  obp = ("obp", "mean"),
                  slg = ("slg", "mean"),
                  bb_rate = ("bb_rate", "mean"),
                  k_rate = ("k_rate", "mean"),
              )
)

cluster_summary["Cluster"] = cluster_summary["GMM_Cluster"].map({
    0: "🔴",
    1: "🔵",
    2: "🟢",
    3: "🟠",
})

cluster_summary["bat_speed"] = cluster_summary["bat_speed"].round(1)
cluster_summary["attack_angle"] = cluster_summary["attack_angle"].round(1)
cluster_summary["vba"] = cluster_summary["vba"].round(1)
cluster_summary["ttc"] = cluster_summary["ttc"].round(3)
cluster_summary["swing_length"] = cluster_summary["swing_length"].round(1)
cluster_summary["avg"] = cluster_summary["avg"].round(3)
cluster_summary["obp"] = cluster_summary["obp"].round(3)
cluster_summary["slg"] = cluster_summary["slg"].round(3)
cluster_summary["bb_rate"] = cluster_summary["bb_rate"].round(1)
cluster_summary["k_rate"] = cluster_summary["k_rate"].round(1)

cluster_summary = cluster_summary.rename(
    columns={
        "bat_speed": "Bat Speed",
        "attack_angle": "Attack Angle",
        "vba": "VBA",
        "ttc": "TTC",
        "swing_length": "Swing Length",
        "avg": "AVG",
        "obp": "OBP",
        "slg": "SLG",
        "bb_rate": "BB%",
        "k_rate": "K%",
    }
)

cluster_summary = cluster_summary[[
    "Cluster",
    "Bat Speed",
    "Attack Angle",
    "VBA",
    "TTC",
    "Swing Length",
    "AVG",
    "OBP",
    "SLG",
    "BB%",
    "K%",
]]

# Load the GMM model
gmm_path = Path(__file__).resolve().parent.parent / "models" / "gmm.joblib"
gmm = joblib.load(gmm_path)

### Create swing shape plot
# only show lines when building/troubleshooting app, remove for real use
show_lines = False

def swing_shape_plotting(id, side, show_hard_hit):
    mode = "lines+markers" if show_lines else "markers"

    df = swing_shape_df[swing_shape_df["batter"] == id]

    # Default to RHB for switch hitters
    if side == "":
        if (df["side"] == "R").any() and (df["side"] == "L").any():
            side = "R"
            actual_side = "R"

    # switch hitter handling
    if side != "":
        df = df[df["side"] == side]
    
    # before anything is selected
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=None,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig
    
    # Get the mean batter distance from the plate and mean depth in the box
    if side == "":
        mean_dist = swing_shape_df["batter_distance"].mean()
        mean_depth = swing_shape_df["batter_depth"].mean()
        actual_side = df["side"].iloc[0]
    else:
        mean_dist = swing_shape_df.loc[swing_shape_df["side"] == side, "batter_distance"].mean()
        mean_depth = swing_shape_df.loc[swing_shape_df["side"] == side, "batter_depth"].mean()
        actual_side = side
        
    # Assign distance and depth based on above or below mean
    contact_side = df["batter_distance"].iloc[0] - mean_dist
    depth_offset = df["batter_depth"].iloc[0] - mean_depth
    
    # create 3d visual
    fig = go.Figure()

    # home plate figure
    x = [0, 0, -8.5, -17, -8.5]
    y = [-8.5, 8.5, 8.5, 0.0, -8.5]
    z = [0, 0, 0, 0, 0]

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
                  hoverinfo="skip",
                  )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [0, 0, -8.5, -17.0, -8.5, 0],
            y = [-8.5, 8.5, 8.5, 0.0, -8.5, -8.5],
            z = [0, 0, 0, 0, 0, 0],
            mode="lines",
            line=dict(color="white", width=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    zone_top = df.iloc[0]["zone_top"]
    zone_bot = df.iloc[0]["zone_bot"]

    # Strike zone figure outline
    fig.add_trace(
        go.Scatter3d(
            x = [0, 0, 0, 0, 0],
            y = [-8.5, -8.5, 8.5, 8.5, -8.5],
            z = [zone_top, zone_bot, zone_bot, zone_top, zone_top],
            mode = "lines",
            line = dict(
                color = "white",
                width = 6,
            ),
            showlegend = False,
            hoverinfo="skip",
        )
    )

    # Strike zone figure fill
    fig.add_trace(
        go.Mesh3d(
            x = [0, 0, 0, 0, 0],
            y = [-8.5, -8.5, 8.5, 8.5, -8.5],
            z = [zone_top, zone_bot, zone_bot, zone_top, zone_top],

            i = [0, 0],
            j = [1, 2],
            k = [2, 3],

            color = "white",
            opacity = 0.15,
            flatshading = True,
            showscale = False,
            hoverinfo="skip",
        )
    )

    # Add hard hit balls if toggled, fly balls and line drives only
    if show_hard_hit:
        hh = pbp_df[(pbp_df["BatterID"] == id) 
                    & (pbp_df["HardHit"] == 1.0)
                    & ((pbp_df["isLD"] == 1)
                    | (pbp_df["isFB"] == 1))
                    & (
                        ((actual_side == "R") & (pbp_df["BatterHand"] == 0))
                        | ((actual_side == "L") & (pbp_df["BatterHand"] == 1))
                    )
                    ][["PitchX", "PitchZ"]]

        if not hh.empty:
            fig.add_trace(
            go.Scatter3d(
                x=np.zeros(len(hh)),          
                y=hh["PitchX"],               
                z=hh["PitchZ"],               
                mode="markers",
                marker=dict(
                    color="red",
                    size=5,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
            )

    vba = df['vba'].iloc[0]

    # Determine depth and height of the contact point, assuming the hands are in-line with the top of the zone
    # when the batter's distance from the plate is the mean, their hands are in line with the front of the plate
    con_depth = df['contact_depth'].iloc[0]
    con_height = zone_top + 31 * np.sin(np.radians(vba))

    # Plot the baseball and contact point
    r = 1.45
    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]

    # If it's a righty, flip the side-coordinate to be negative
    flip = 1
    if actual_side == "R":
        flip = -1

    fig.add_trace(
        go.Surface(
            x = con_depth + r*np.cos(u)*np.sin(v),
            y = flip * (contact_side) + r*np.sin(u)*np.sin(v),
            z = con_height + r*np.cos(v),
            colorscale=[[0, "white"], [1, "white"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    ### Plot the bat, 34 inch bat
    ### Plot the bat, 34 inch bat
    bat_bottom = 31
    bat_top = 3

    # Contact point, adjust so the ball and bat aren't overlapping
    ball_point = np.array([con_depth, flip * contact_side, con_height])
    ball_point[0] -= 2.5

    attack_angle = df["attack_angle"].iloc[0]

    if actual_side == "R":
        attack_angle = -attack_angle
    else:
        attack_angle = attack_angle

    aa = np.radians(attack_angle)

    dir_x = -np.sin(aa) if actual_side == "R" else np.sin(aa)
    dir_y = flip * np.cos(aa)
    dir_z = np.tan(np.radians(vba))

    bat_dir = np.array([dir_x, dir_y, dir_z])
    bat_dir = bat_dir / np.linalg.norm(bat_dir)

    knob_point = ball_point - bat_dir * bat_bottom
    barrel_end = ball_point + bat_dir * bat_top

  #  knob_side_end = ball_point - bat_dir * r
  #  barrel_side_start = ball_point + bat_dir * r

    def draw_bat(fig, knob_point, barrel_end):
        knob_point = np.asarray(knob_point, dtype=float)
        barrel_end = np.asarray(barrel_end, dtype=float)

        axis = barrel_end - knob_point
        L = np.linalg.norm(axis)
        axis /= L

        # Build perpendicular basis
        tmp = np.array([0., 0., 1.]) if abs(axis[2]) < 0.95 else np.array([1., 0., 0.])
        u = np.cross(axis, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        theta = np.linspace(0, 2*np.pi, 48)
        s = np.linspace(0, 1, 80)

        X = np.zeros((len(s), len(theta)))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)

        for i, t in enumerate(s):
            center = knob_point + t * (barrel_end - knob_point)

            d = t * L  # inches from knob

            # Radius profile (inches)
            if d < 0.18:
                r = 0.88
            elif d < 0.6:
                t = (d - 0.18) / 0.42
                r = 0.88 - (0.88 - 0.65) * (3*t**2 - 2*t**3)
            elif d < 10:
                r = 0.65 - (0.65 - 0.52) * (d - 1) / 9
            elif d < 28:
                t = (d - 10) / 18
                r = 0.52 + (1.60 - 0.52) * (3*t**2 - 2*t**3)
            elif d < L - 0.30:
                r = 1.60
            else:
                t = (d - (L - 0.30)) / 0.30
                r = 1.60 * (1 - 0.15 * (3*t**2 - 2*t**3))
            

            circle = (
                center[:, None]
                + r * (
                    u[:, None] * np.cos(theta)
                    + v[:, None] * np.sin(theta)
                )
            )

            X[i] = circle[0]
            Y[i] = circle[1]
            Z[i] = circle[2]

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=[[0, "#C68642"], [1, "#C68642"]],
                showscale=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.90,
                    specular=0.35,
                    roughness=0.55,
                ),
                lightposition=dict(x=100, y=100, z=200),
                hoverinfo="skip",
            )
        )

        # Knob
        theta = np.linspace(0, 2*np.pi, 60)
        R = 0.88

        disk = (
            knob_point[:, None]
            + R * (u[:, None] * np.cos(theta) + v[:, None] * np.sin(theta))
        )

        fig.add_trace(
            go.Mesh3d(
                x=np.r_[knob_point[0], disk[0]],
                y=np.r_[knob_point[1], disk[1]],
                z=np.r_[knob_point[2], disk[2]],
                i=[0]*len(theta),
                j=np.arange(1, len(theta)+1),
                k=np.r_[np.arange(2, len(theta)+1), 1],
                intensity=np.ones(len(theta) + 1),
                colorscale = [[0, "#C68642"], [1, "#C68642"]],
                flatshading = True,
                showscale = False,
                hoverinfo="skip",
            )
        )   

        # Barrel
        theta = np.linspace(0, 2*np.pi, 60)
        R = 1.39

        cap = (
            barrel_end[:, None]
            + R * (u[:, None] * np.cos(theta) + v[:, None] * np.sin(theta))
            )

        fig.add_trace(
            go.Mesh3d(
                x=np.r_[barrel_end[0], cap[0]],
                y=np.r_[barrel_end[1], cap[1]],
                z=np.r_[barrel_end[2], cap[2]],
                i=[0]*len(theta),
                j=np.arange(1, len(theta)+1),
                k=np.r_[np.arange(2, len(theta)+1), 1],
                intensity=np.ones(len(theta) + 1),
                colorscale = [[0, "#C68642"], [1, "#C68642"]],
                flatshading = True,
                showscale = False,
                hoverinfo="skip",
            )
        )

    draw_bat(fig, knob_point, barrel_end)

    ### Draw the swing shape, adapted from the old app
    # Convert swing length to inches
    swing_len = df["swing_length"].iloc[0]
    swing_len = swing_len * 12

    # Batter height is about top of the zone * 1.82 assuming belt line is about 55% of height
    hand_height = 1.82 * zone_top

    # Batter distance is given, needs to be negative for RHB
    batter_distance = df["batter_distance"].iloc[0] * flip

    # Same direction as the bat
    x_dir, y_dir, z_dir = bat_dir

    # Swing shape start point (Contact)
    swing_shape_starting_x = ball_point[0]
    swing_shape_starting_y = ball_point[1]
    swing_shape_starting_z = ball_point[2]

    # Swing shape end point (Launch position), bat length is 34, sweet spot around 31
    # Front of plate, adjusted for depth offset, minus swing length * 2/3
    swing_shape_ending_x = depth_offset - swing_len * (2/3)

    swing_shape_ending_y = -1 * (batter_distance + 34 * y_dir)

    swing_shape_ending_z = hand_height + 34 * z_dir

    # Swing shape middle point
    swing_shape_middle_x = depth_offset - swing_len
    swing_shape_middle_y = 6 * y_dir
    swing_shape_middle_z = hand_height + 31 - hand_height * 1.2

    # Draw the swing shape curve
    # Control points
    P0 = np.array([swing_shape_starting_x, swing_shape_starting_y, swing_shape_starting_z])
    P1 = np.array([swing_shape_middle_x, swing_shape_middle_y, swing_shape_middle_z])
    P2 = np.array([swing_shape_ending_x, swing_shape_ending_y, swing_shape_ending_z])

    # Quadratic Bézier curve
    t = np.linspace(0, 1, 100)

    curve = (
        (1 - t)[:, None]**2 * P0
        + 2 * (1 - t)[:, None] * t[:, None] * P1
        + t[:, None]**2 * P2
    )

    fig.add_trace(
        go.Scatter3d(
            x=curve[:, 0],
            y=curve[:, 1],
            z=curve[:, 2],
            mode="lines",
            line=dict(color="lightblue", width=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            dragmode = "orbit",
            xaxis=dict(
                title="",
                visible=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,
            ),
            yaxis=dict(
                title="",
                visible=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,
            ),
            zaxis=dict(
                title="",
                visible=False,
                showgrid=False,
                zeroline=False,
                showbackground=False,
            ),
            aspectmode="data",
        ),
    )

    return fig

### Swing Classification Plotting function
# Mapping for variable names
cluster_columns = {
    "Bat Speed": "bat_speed",
    "Attack Angle": "attack_angle",
    "Vertical Bat Angle (VBA)": "vba",
    "Time to Contact (TTC)": "ttc",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "BB%": "bb_rate",
    "K%": "k_rate",
}
        
def swing_cluster_plot(x_name, y_name):

    x_col = cluster_columns[x_name]
    y_col = cluster_columns[y_name]

    colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "orange",
    }

    fig = go.Figure()

    for cluster, color in colors.items():

        df = cluster_df[cluster_df["GMM_Cluster"] == cluster]

        fig.add_trace(
            go.Scatter(
                x = df[x_col],
                y = df[y_col],
                mode = "markers",
                marker = dict(
                    color = color,
                    size = 8,
                ),
                hoverinfo = "skip",
                showlegend = False,
            )
        )

        fig.update_layout(
            template = "plotly_dark",
            height = 500,
            xaxis_title = x_name,
            yaxis_title = y_name,
            margin = dict(l = 20, r = 20, t = 20, b = 20)
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
    show_hard_hit: str = "No"

    # Swing classification state variables
    cluster_x: str = "Bat Speed"
    cluster_y: str = "Attack Angle"
    
    # Default classification model inputs
    bat_speed_input: float = 70.0
    attack_angle_input: float = 10.0
    vba_input: float = 30.0
    ttc_input: float = 0.14

    def set_landing(self):
        self.current_tab = "Landing"
    
    def set_model_visual(self):
        self.current_tab = "Model Visuals"

    def set_swingshape(self):
        self.current_tab = "Swing Shape"

    def set_swingclass(self):
        self.current_tab = "Swing Classification"

    # Functions for swing shape tab
    def set_selected_batter(self, name: str):
        self.selected_batter = int(swing_shape_df.loc[swing_shape_df["name"] == name, "batter"].iloc[0])
        self.selected_side = ""

    def set_selected_side(self, side: str):
        self.selected_side = side

    def set_show_hard_hit(self, value: str):
        self.show_hard_hit = value

    def get_stat(self, column: str, decimals: int | None = None) -> str:
        if self.selected_batter == 0:
            return ""
        
        rows = swing_shape_df[swing_shape_df["batter"] == self.selected_batter]

        if len(rows) > 1:
            if self.selected_side == "":
                return ""
            side = "R" if self.selected_side == "RHB" else "L"
            rows = rows[rows["side"] == side]

        value = rows.iloc[0][column]

        if decimals is None:
            return str(value)
        
        return f'{value:.{decimals}f}'
    
    # Functions for swing classification tab
    def set_cluster_x(self, value: str):
        self.cluster_x = value
    
    def set_cluster_y(self, value: str):
        self.cluster_y = value

    def set_bat_speed_input(self, value):
        self.bat_speed_input = float(value)

    def set_attack_angle_input(self, value):
        self.attack_angle_input = float(value)

    def set_vba_input(self, value):
        self.vba_input = float(value)

    def set_ttc_input(self, value):
        self.ttc_input = float(value)
    
    # Get the swing metrics to list in the swing shape visualizer page
    @rx.var
    def bat_speed(self) -> str:
        return self.get_stat("bat_speed", 1)
    
    @rx.var
    def attack_angle(self) -> str:
        return self.get_stat("attack_angle", 1)
    
    @rx.var
    def vba(self) -> str:
        return self.get_stat("vba", 1)
    
    @rx.var
    def ttc(self) -> str:
        return self.get_stat("ttc")
    
    @rx.var
    def swing_length(self) -> str:
        return self.get_stat("swing_length", 1)
    
    @rx.var
    def avg(self) -> str:
        return self.get_stat("avg", 3)
    
    @rx.var
    def obp(self) -> str:
        return self.get_stat("obp", 3)
    
    @rx.var
    def slg(self) -> str:
        return self.get_stat("slg", 3)
    
    @rx.var
    def bb_rate(self) -> str:
        return self.get_stat("bb_rate", 1)
    @rx.var
    def k_rate(self) -> str:
        return self.get_stat("k_rate", 1)

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
    
    # Cluster summary
    @rx.var
    def cluster_summary(self) -> list[dict]:
        return cluster_summary.to_dict("records")
    
    # Swing figure
    @rx.var
    def swing_fig(self) -> go.Figure:
        side = "R" if self.selected_side == "RHB" else "L" if self.selected_side == "LHB" else ""
        return swing_shape_plotting(self.selected_batter, side, self.show_hard_hit == "Yes", )
    
    # Swing classification figure
    @rx.var
    def cluster_fig(self) -> go.Figure:
        return swing_cluster_plot(self.cluster_x, self.cluster_y)
    
    @rx.var
    def cluster_axis_options(self) -> list[str]:
        return list(cluster_columns.keys())
    
    # Function for assigning user cluster probabilities
    @rx.var
    def cluster_probs(self) -> list[float]:
        X = [[
            self.bat_speed_input,
            self.attack_angle_input,
            self.vba_input,
            self.ttc_input,
        ]]

        probs = gmm.predict_proba(X)[0]

        return [round(100 * p, 1) for p in probs]

    
### Side bar for tab selection
def sidebar():
    return rx.vstack(rx.heading("Tab Options", size = "5"),
                     rx.divider(),
                     
                     rx.button("Landing", width = "100%", on_click = AppState.set_landing, ),
                     rx.button("Expected Performance", width = "100%", on_click = AppState.set_model_visual, ),
                     rx.button("Swing Shape Visualization", width = "100%", on_click = AppState.set_swingshape, ),
                     rx.button("Swing Classification", width = "100%", on_click = AppState.set_swingclass, ),
                     
                     width = "220px",
                     min_height = "100vh",
                     align_self = "stretch",
                     padding = "1em",
                     spacing = "4",
                     border_right = "1px solid lightgray",
                     )

### Landing page
def landing_tab():
    return rx.vstack(
        # Main title
        rx.heading("Swing Shape Analysis Tool", size = "8"),
        rx.text(""),

        # subsection
        rx.heading("About"),

        rx.text("Welcome to the Swing Shape Analysis Tool. Here, you can use data gathered by bat sensors to explore how bat tracking data and swing mechanics relate to"),
        rx.text("performance on the field."),
        rx.text(""),
        rx.text(""),
        rx.text("This project was inspired by MLB's public release of bat tracking data. I owned a bat sensor during my playing career so that I could keep track of my swing data,"),
        rx.text("but never had enough data to understand what each metric meant for my actual hitting performance. With the release of MLB's bat tracking data, in-depth analysis that"),
        rx.text("compares bat tracking data to on-field performance is now possible. This tool aims to provide users with a comprehensive analysis of their swing data and a roadmap for"),
        rx.text("maximizing their ability."),

        rx.text(""),
        rx.text(""),

        # Definitions
        rx.heading("Metric definitions", size = "5"),
        rx.heading("Definitions provided by mlb.com", size = "2"),

        rx.text(""),
        rx.text(""),

        rx.text("Bat Speed (MPH): How fast the sweet spot of the bat is moving, in mph, at the point of contact with the ball (or where the ball and bat would have met, in"),
        rx.text("case of a swing-and-miss)"),

        rx.text(""),

        rx.text("Attack Angle (deg): The vertical direction that the sweet spot of the bat is traveling at the moment it hits the baseball"),

        rx.text(""),

        rx.text("Swing Length (ft): Captured from the start of a swing until impact point."),

        rx.text(""),

        rx.heading("Definitions provided by blastmotion.com", size = "2"),

        rx.text(""),

        rx.text("Vertical Bat Angle (deg): The angle of the bat relative to horizontal at impact. Zero means the barrel and knob are parallel to the ground. Negative means the"),
        rx.text("barrel is below the knob."),

        rx.text(""),

        rx.text("Time to Contact (s): Elapsed time from the start of the downswing to impact. A time below 0.14 seconds is a benchmark associated with elite prep-level tools."),
    )

### Model visual page
def model_visual_tab():
    return rx.vstack(rx.heading("Model Visuals"), rx.text("Model Visuals"), align_items = "start", )

### Tab for seeing MLB swing shapes
def swing_shape_tab():

    return rx.vstack(
        # Swing Shape module title
        rx.heading("Swing Shape Visualizer", size = "8", ), 
        
        # Top row
        rx.hstack(

            # Left side
            rx.vstack(
                
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
                        rx.select(
                            ["No", "Yes"],
                            placeholder = "View Hard Hit Balls",
                            value = AppState.show_hard_hit,
                            on_change = AppState.set_show_hard_hit,
                            width = "200px",
                        ),

                        align_items = "start",
                        spacing = "4",
            ),

            rx.spacer(),

            # Right side
            rx.cond(
                AppState.bat_speed != "",
                    rx.vstack(
                    rx.text(f"Bat Speed (MPH): {AppState.bat_speed}"),
                    rx.text(f"Attack Angle (deg): {AppState.attack_angle}"),
                    rx.text(f"Vertical Bat Angle (VBA) (deg): {AppState.vba}"),
                    rx.text(f"Time to Contact (TTC) (s): {AppState.ttc}"),
                    rx.text(f"Swing Length (ft): {AppState.swing_length}"),
                    align_items="start",
                    spacing="2",
                    margin_left="400px",
                    ),
                ),
            ),

        # Swing shape plot
        rx.center(
            rx.plotly(
                data = AppState.swing_fig,
                width = "95%",
                height = "450px",
                config = {
                    "displayModeBar": False,
                },
            ),
            width = "100%",
        ),

        rx.cond(
            AppState.avg != "",
            rx.text(
                f"AVG: {AppState.avg} / "
                f"OBP: {AppState.obp} / "
                f"SLG: {AppState.slg} / "
                f"BB%: {AppState.bb_rate} / "
                f"K%: {AppState.k_rate}",
                font_size="14px",
            ),
        ),

        align_items = "start",
        spacing = "5",
        
    )

### Swing Classification Page
def swing_classification_tab():
    return rx.vstack(

        # Swing Classification Title
        rx.heading("Swing Classification"),

        rx.hstack(
            
            rx.select(
                items = AppState.cluster_axis_options,
                value = AppState.cluster_x,
                on_change = AppState.set_cluster_x,
                width = "260px",
            ),

            rx.select(
                items = AppState.cluster_axis_options,
                value = AppState.cluster_y,
                on_change = AppState.set_cluster_y,
                width = "260px",
            ),

            spacing = "4",
        ),

        rx.plotly(
            data = AppState.cluster_fig,
            width = "900px",
            height = "500px",
            config = {
                "displayModeBar": False,
            },
        ),

        rx.hstack(
            rx.vstack(
            rx.text("Notable members of each cluster: ", style = {"fontSize": "12px"}),
            rx.text("🔴: James Wood (L), Aaron Judge (R)", style = {"fontSize": "11px"}),
            rx.text("🔵: Kyle Schwarber (L), Mike Trout (R)", style = {"fontSize": "11px"}),
            rx.text("🟢: Jonathan Aranda (L), Mookie Betts (R)", style = {"fontSize": "11px"}), 
            rx.text("🟠: Luis Arráez (L), Jacob Wilson (R)", style = {"fontSize": "11px"}),
            width="240px",
            align_items="start",
            spacing="4",
            flex_shrink="0",
            ),

        # Summary table
        rx.box(
            rx.table.root(

                rx.table.header(
                    rx.table.row(
                        *[
                            rx.table.column_header_cell(h)
                            for h in [
                                "Cluster",
                                "Bat Speed",
                                "Attack Angle",
                                "VBA",
                                "TTC",
                                "Swing Length",
                                "AVG",
                                "OBP",
                                "SLG",
                                "BB%",
                                "K%",
                            ]
                        ]
                    ),
                    style = {"fontSize": "12px"},
                ),

            rx.table.body(
                    rx.foreach(
                        AppState.cluster_summary,
                        lambda row: rx.table.row(

                            rx.table.cell(row["Cluster"]),
                            rx.table.cell(row["Bat Speed"]),
                            rx.table.cell(row["Attack Angle"]),
                            rx.table.cell(row["VBA"]),
                            rx.table.cell(row["TTC"]),
                            rx.table.cell(row["Swing Length"]),
                            rx.table.cell(row["AVG"]),
                            rx.table.cell(row["OBP"]),
                            rx.table.cell(row["SLG"]),
                            rx.table.cell(row["BB%"]),
                            rx.table.cell(row["K%"]),
                        ),
                    ),
                    style = {"fontSize": "12px"},
                ),

            ),
            flex = "1",
            overflow_x = "auto",
            ),

            width = "100%",
            align_items = "start",
            spacing = "5",
        ),

        rx.heading("Predict Swing Cluster"),

        rx.input(
            value = AppState.bat_speed_input,
            on_change = AppState.set_bat_speed_input,
            placeholder = "Bat Speed",
        ),

        rx.input(
            value = AppState.attack_angle_input,
            on_change = AppState.set_attack_angle_input,
            placeholder = "Attack Angle",
        ),

        rx.input(
            value = AppState.vba_input,
            on_change = AppState.set_vba_input,
            placeholder = "Vertical Bat Angle",
        ),

        rx.input(
            value = AppState.ttc_input,
            on_change = AppState.set_ttc_input,
            placeholder = "Time to Contact",
        ),

        rx.vstack(
            rx.text(f"🔴 Cluster 0: {AppState.cluster_probs[0]}%"),
            rx.text(f"🔵 Cluster 1: {AppState.cluster_probs[1]}%"),
            rx.text(f"🟢 Cluster 2: {AppState.cluster_probs[2]}%"),
            rx.text(f"🟠 Cluster 3: {AppState.cluster_probs[3]}%"),
            align_items="start",
            ),
    )


### Main content
def content():
    return rx.match(
        AppState.current_tab,
        ("Landing", landing_tab()),
        ("Model Visuals", model_visual_tab()),
        ("Swing Shape", swing_shape_tab()),
        ("Swing Classification", swing_classification_tab()),
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
                     min_height = "100vh",
                     spacing = "0",
                     )

app = rx.App()
app.add_page(index, route = "/")