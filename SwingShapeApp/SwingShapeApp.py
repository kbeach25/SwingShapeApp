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

# Create dropdowns for pitch types and other filters
# pitch type
pitch_type_options = [
    "FF", "SI", "FC",
    "SL", "CU", 
    "CH", "FS", 
]

# pitcher hand
pitcher_hand_options = ["R", "L"]

# batter hand
batter_hand_options = ["R", "L"]

# targets
target_options = ["Contact", "Hard Hit", "Ground Ball", "Line Drive", "Fly Ball", "Pop Up",]

# pitch velocity
slider_min = int(np.ceil(pbp_df["ReleaseSpeed"].min()))
slider_max = int(np.floor(pbp_df["ReleaseSpeed"].max()))
default_speed = (slider_min + slider_max) // 2

# pitch type family dictionary
pitch_map = {
    "FF": 0,
    "SI": 0,
    "FC": 0,
    "SL": 1,
    "CU": 1,
    "CH": 2,
    "FS": 2,
}

pitch_specific_map ={
    "FF": 0,
    "SI": 1,
    "FC": 2,
    "SL": 3,
    "CU": 4,
    "CH": 5,
    "FS": 6,
    "ST": 7,
}

# parts of the strike zone
valid_zones = [
    int(f'{row}{col}')
    for row in range(1, 9)
    for col in range(1, 9)
]

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

# Load the GMM model and scaler, only needed for predicting a user's swing cluster
gmm_path = Path(__file__).resolve().parent.parent / "models" / "gmm.joblib"
gmm = joblib.load(gmm_path)

scaler_path = Path(__file__).resolve().parent.parent / "models" / "gmm_scaler.joblib"
gmm_scaler = joblib.load(scaler_path)

# Load contact, hard hit, and batted ball models
contact_model_path = Path(__file__).resolve().parent.parent / "models" / "contact_xgb_model.pkl"
contact_model = joblib.load(contact_model_path)

hard_hit_model_path = Path(__file__).resolve().parent.parent / "models" / "hard_hit_model.pkl"
hard_hit_model = joblib.load(hard_hit_model_path)

bb_model_path = Path(__file__).resolve().parent.parent / "models" / "bb_model.pkl"
bb_model = joblib.load(bb_model_path)

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

    # Batter height is about top of the zone * 2 assuming belt line is about 50% of height
    hand_height = 2 * zone_top

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
        scene = dict(
            camera = dict(
            eye = dict(
                x = 0,
                y = -2.8 if actual_side == "R" else 2.8,
                z = 0,
                ),
            ),
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

prob_columns = {
    "Contact": "pContact",
    "Hard Hit": "pHardHit",
    "Ground Ball": "pGB",
    "Line Drive": "pLD",
    "Fly Ball": "pFB",
    "Pop Up": "pPU",
}

def strike_zone_plot(probs, target):
    fig = go.Figure()

    if len(probs) != 64:
        fig.update_layout(
            template="plotly_dark",
            height=500,
            width=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    grid = np.array(probs).reshape(8, 8)

    target_centers = {
        "Contact": 0.750,
        "Hard Hit": 0.410,
        "Ground Ball": 0.40,
        "Line Drive": 0.25,
        "Fly Ball": 0.25,
        "Pop Up": 0.10,
    }

    center = target_centers.get(target, 0.50)

    vmin = float(grid.min())
    vmax = float(grid.max())

    if vmin < center < vmax:
        zmid = center
    else:
        zmid = (vmin + vmax) / 2

    # create text labels
    text = [
        [f"{100 * value:.1f}%" for value in row]
        for row in grid
    ]

    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=list(range(1, 9)),
            y=list(range(8, 0, -1)),
            zmin=vmin,
            zmax=vmax,
            zmid=center,
            colorscale="RdBu_r",
            text=text,
            texttemplate="%{text}",
            textfont=dict(
                size=11,
                color="black",
            ),
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Predicted {target} Probability",
            x=0.46,
            xanchor="center",
        ),
        width=500,
        height=600,
        margin=dict(l=40, r=80, t=70, b=40),

        xaxis=dict(
            visible=False,
            range=[0.5, 8.5],
            constrain="domain",
        ),

        yaxis=dict(
            visible=False,
            range=[0.5, 8.5],
            scaleanchor="x",
            scaleratio=1.4,
        ),

        plot_bgcolor="white",
    )

    # outside border
    fig.add_shape(
        type="rect",
        x0=0.5,
        x1=8.5,
        y0=0.5,
        y1=8.5,
        line=dict(
            color="black",
            width=3,
        ),
    )

    return fig

### App state class
class AppState(rx.State):
    # Landing page
    current_tab: str = "Landing"

    # State variables

    # Strike zone visual tab variables
    personal_mode: bool = False
    player_mode: bool = False

    def show_personal_mode(self):
        self.personal_mode = True
        self.player_mode = False

    def show_player_mode(self):
        self.player_mode = True
        self.personal_mode = False

    target: str = "Contact"
    selected_pitch_type: str = ""
    pitch_velocity_min: int = max(slider_min, default_speed - 3)
    pitch_velocity_max: int = min(slider_max, default_speed + 3)
    pitcher_hand: str = "R"
    batter_hand: str = "R"
    model_bat_speed: str = ""
    model_attack_angle: str = ""
    model_vba: str = ""
    model_ttc: str = ""
    predicted_zone_probs: list[float] = []

    def predict_zone(self):
        if self.pitch_velocity_min > self.pitch_velocity_max:
            return

        try:
            bat_speed = float(self.model_bat_speed)
            attack_angle = float(self.model_attack_angle)
            vba = float(self.model_vba)
            ttc = float(self.model_ttc)

        except ValueError:
            return
        
        if not (60 <= bat_speed <= 90):
            return

        if not (-20 <= attack_angle <= 20):
            return

        if not (-50 <= vba <= -1):
            return

        if not (0.120 <= ttc <= 0.160):
            return

        if self.selected_pitch_type == "":
            return

        pitch_type = pitch_map.get(self.selected_pitch_type)
        pitch_type_specific = pitch_specific_map.get(self.selected_pitch_type)

        if pitch_type is None or pitch_type_specific is None:
            return

        # handedness encoding
        batter_hand = 0 if self.batter_hand == "R" else 1
        pitcher_hand = 0 if self.pitcher_hand == "R" else 1

        # midpoint of velocity slider
        pitch_velocity = (self.pitch_velocity_min + self.pitch_velocity_max) / 2

        rows = []

        for zone in valid_zones:
            rows.append({
                "BatSpeed": bat_speed,
                "AttackAngle": attack_angle,
                "VBA": vba,
                "TTC": ttc,
                "ReleaseSpeed": pitch_velocity,
                "PitchType": pitch_type,
                "PitchTypeSpecific": pitch_type_specific,
                "PitcherHand": pitcher_hand,
                "BatterHand": batter_hand,
                "PitchZone": zone,
            })

        pred_df = pd.DataFrame(rows)

        # select model
        if self.target == "Contact":
            probs = contact_model.predict_proba(pred_df)[:, 1]

        elif self.target == "Hard Hit":
            probs = hard_hit_model.predict_proba(pred_df)[:, 1]

        else:
            probs = bb_model.predict_proba(pred_df)

            classes = list(bb_model.named_steps["lr"].classes_)

            target_class = {
                "Ground Ball": "isGB",
                "Line Drive": "isLD",
                "Fly Ball": "isFB",
                "Pop Up": "isPU",
            }[self.target]

            probs = probs[:, classes.index(target_class)]

        self.predicted_zone_probs = probs.tolist()

    def set_pitch_velocity_min(self, value):
        self.pitch_velocity_min = int(value[0])

    def set_pitch_velocity_max(self, value):
        self.pitch_velocity_max = int(value[0])
    

    # Swing shape module variables
    selected_batter: int = 0
    selected_side: str = ""
    show_hard_hit: str = "No"

    # Swing classification state variables
    cluster_x: str = "Bat Speed"
    cluster_y: str = "Attack Angle"
    
    # Default classification model inputs
    bat_speed_input: str = "70.0"
    attack_angle_input: str = "10.0"
    vba_input: str = "-30.0"
    ttc_input: str = "0.14"

    prediction_error: str = ""
    predicted_probs: list[float] = [0.0, 0.0, 0.0, 0.0]
    prediction_ready: bool = False

    def set_target(self, value):
        self.target = value

    def set_selected_pitch_type(self, value):
        self.selected_pitch_type = value

        pitch_code = pitch_specific_map.get(value)

        if pitch_code is None:
            return

        speeds = pbp_df.loc[
            pbp_df["PitchTypeSpecific"] == pitch_code,
            "ReleaseSpeed"
        ]

        if speeds.empty:
            return

        new_min = int(np.ceil(speeds.min()))
        new_max = int(np.floor(speeds.max()))

        self.pitch_velocity_min = new_min
        self.pitch_velocity_max = new_max

    def set_pitcher_hand(self, value):
        self.pitcher_hand = value

    def set_batter_hand(self, value):
        self.batter_hand = value

    def set_model_bat_speed(self, value):
        self.model_bat_speed = value

    def set_model_attack_angle(self, value):
        self.model_attack_angle = value

    def set_model_vba(self, value):
        self.model_vba = value

    def set_model_ttc(self, value):
        self.model_ttc = value

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
        self.bat_speed_input = value

    def set_attack_angle_input(self, value):
        self.attack_angle_input = value

    def set_vba_input(self, value):
        self.vba_input = value

    def set_ttc_input(self, value):
        self.ttc_input = value

    @rx.var
    def pitch_speed_min(self) -> int:
        if self.selected_pitch_type == "":
            return slider_min

        pitch_code = pitch_specific_map.get(self.selected_pitch_type)

        if pitch_code is None:
            return slider_min

        speeds = pbp_df.loc[
            pbp_df["PitchTypeSpecific"] == pitch_code,
            "ReleaseSpeed"
        ]

        if speeds.empty:
            return slider_min

        return int(np.ceil(speeds.min()))


    @rx.var
    def pitch_speed_max(self) -> int:
        if self.selected_pitch_type == "":
            return slider_max

        pitch_code = pitch_specific_map.get(self.selected_pitch_type)

        if pitch_code is None:
            return slider_max

        speeds = pbp_df.loc[
            pbp_df["PitchTypeSpecific"] == pitch_code,
            "ReleaseSpeed"
        ]

        if speeds.empty:
            return slider_max

        return int(np.floor(speeds.max()))

    # variable for strike zone
    @rx.var
    def predicted_zone_fig(self) -> go.Figure:
        return strike_zone_plot(
            self.predicted_zone_probs,
            self.target,
        )
    
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
    def predict_cluster(self):
        try:
            bat_speed = float(self.bat_speed_input)
            attack_angle = float(self.attack_angle_input)
            vba = float(self.vba_input)
            ttc = float(self.ttc_input)
        except ValueError:
            self.prediction_error = "Please enter valid values. Bat Speed and TTC must be positive, VBA must be negative."
            self.prediction_ready = False
            return
        
        if bat_speed <= 0 or ttc <= 0 or vba > 0:
            self.prediction_error = "Please enter valid values. Bat Speed and TTC must be positive, VBA must be negative."
            self.prediction_ready = False
            return
        
        X = [[bat_speed, attack_angle, vba, ttc]]

        X_scaled = gmm_scaler.transform(X)
        probs = gmm.predict_proba(X_scaled)[0]

        self.predicted_probs = [round(100 * p, 1) for p in probs]
        self.prediction_error = ""
        self.prediction_ready = True

    
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

# function for personal strike zone visual
def personal_visualizer():

    return rx.hstack(

        # INPUTS
        rx.vstack(

            rx.heading("Swing Inputs", size="4"),

            rx.text("Bat Speed (mph): 60 to 90 mph"),
            rx.input(
                type = "number",
                placeholder="70.0",
                min = 60,
                max = 90,
                step = 0.1,
                value=AppState.model_bat_speed,
                on_change=AppState.set_model_bat_speed,
                width="250px",
            ),

            rx.text("Attack Angle (°): -20° to 20°"),
            rx.input(
                type = "number",
                placeholder="10.0",
                min = -20,
                max = 20,
                step = 0.1,
                value=AppState.model_attack_angle,
                on_change=AppState.set_model_attack_angle,
                width="250px",
            ),

            rx.text("Vertical Bat Angle (VBA) (°): -50° to -1°"),
            rx.input(
                type = "number",
                placeholder="-30.0",
                min = -50,
                max = -1,
                step = 0.1,
                value=AppState.model_vba,
                on_change=AppState.set_model_vba,
                width="250px",
            ),

            rx.text("Time to Contact (s): 0.120 to 0.160 s"),
            rx.input(
                type = "number",
                placeholder="0.140",
                min = 0.120,
                max = 0.160,
                step = 0.001,
                value=AppState.model_ttc,
                on_change=AppState.set_model_ttc,
                width="250px",
            ),

            rx.text("Batter Hand"),
            rx.select(
                batter_hand_options,
                value=AppState.batter_hand,
                on_change=AppState.set_batter_hand,
                width="250px",
            ),

            rx.divider(),

            rx.heading("Pitch Inputs", size="4"),

            rx.text("Pitch Type"),
            rx.select(
                pitch_type_options,
                placeholder="Select pitch type",
                value=AppState.selected_pitch_type,
                on_change=AppState.set_selected_pitch_type,
                width="250px",
            ),

            rx.text("Pitcher Hand"),
            rx.select(
                pitcher_hand_options,
                value=AppState.pitcher_hand,
                on_change=AppState.set_pitcher_hand,
                width="250px",
            ),

            rx.text("Pitch Velocity Range (mph)"),

            rx.text(
                f"{AppState.pitch_velocity_min} - "
                f"{AppState.pitch_velocity_max} mph"
            ),

            rx.text("Minimum Velocity"),

            rx.slider(
                min=AppState.pitch_speed_min,
                max=AppState.pitch_speed_max,
                step=1,
                value=[AppState.pitch_velocity_min],
                on_change=AppState.set_pitch_velocity_min,
                width="250px",
            ),

            rx.text("Maximum Velocity"),

            rx.slider(
                min=AppState.pitch_speed_min,
                max=AppState.pitch_speed_max,
                step=1,
                value=[AppState.pitch_velocity_max],
                on_change=AppState.set_pitch_velocity_max,
                width="250px",
            ),

            rx.text("Prediction"),

            rx.select(
                target_options,
                value=AppState.target,
                on_change=AppState.set_target,
                width="250px",
            ),

            rx.button(
                "Update Visual",
                on_click=AppState.predict_zone,
                width="250px",
            ),

            spacing="3",
            align_items="start",
            width="300px",
        ),

        # VISUAL
        rx.vstack(

            rx.cond(
                AppState.predicted_zone_probs.length() > 0,

                rx.vstack(
                    rx.plotly(
                        data=AppState.predicted_zone_fig,
                        config={
                            "displayModeBar": False,
                            "staticPlot": True,
                        },
                        width="500px",
                    ),

                    rx.text(
                        "Catcher POV",
                        text_align="center",
                        width="100%",
                        font_weight="bold",
                        transform="translateX(-20px)",
                    ),

                    spacing="0",
                    align_items="center",
                    width="500px",
                ),

                rx.box(
                    rx.text(
                        "Enter swing and pitch information, then click Update Visual.",
                        text_align="center",
                    ),
                    width="500px",
                    height="500px",
                    border="2px solid gray",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                ),
            ),

            align_items="center",
        ),

        spacing="8",
        align_items="start",
        width="100%",
    )

### Model visual page
def model_visual_tab():
    return rx.vstack(
        
        rx.heading(
            "Swing Decision Visualizer",
            size = "7",
            ), 
                   
        rx.text(
            "Enter your bat sensor data or choose an MLB player and see how bat sensor data predicts success against pitch types, locations, and velocities.",
            font_size = "15px",
            ),  
            
        rx.hstack(
            rx.button(
                "Enter Personal Swing Data",
                on_click = AppState.show_personal_mode,
            ),

            rx.button(
                "Choose MLB Player",
                on_click = AppState.show_player_mode,
            ),
            
            spacing = "5",
        ),

        rx.cond(
            AppState.personal_mode,

            personal_visualizer(),

            rx.cond(
                AppState.player_mode,
                rx.text("MLB player mode not done"),
            ),
        ),

        align_items = "start",
        spacing = "5",
        width = "100%",           
        )

### Tab for seeing MLB swing shapes
def swing_shape_tab():

    return rx.vstack(
        # Swing Shape module title
        rx.heading("Swing Shape Visualizer", size = "6", ), 
        
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
                        rx.hstack(
                            rx.text("Show Hard Hit Balls?"),
                            rx.select(
                                ["No", "Yes"],
                                value = AppState.show_hard_hit,
                                on_change = AppState.set_show_hard_hit,
                                width = "100px",
                            ),
                            spacing = "3",
                            align_items = "center",
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

        rx.center(
            rx.plotly(
                data=AppState.cluster_fig,
                width="95%",
                height="500px",
                config={
                    "displayModeBar": False,
                },
            ),
            width="100%",
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
                    style={"fontSize": "15px"},
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
                    style={"fontSize": "15px"},
                ),

            ),
            width="100%",
            overflow_x="auto",
        ),

        rx.vstack(
            rx.heading("Notable Members of each Cluster", size="6", margin_top="40px",),

            rx.text("🔴 James Wood (L), Aaron Judge (R)", font_size="20px"),
            rx.text("🔵 Kyle Schwarber (L), Mike Trout (R)", font_size="20px"),
            rx.text("🟢 Jonathan Aranda (L), Mookie Betts (R)", font_size="20px"),
            rx.text("🟠 Luis Arráez (L), Jacob Wilson (R)", font_size="20px"),

            width="100%",
            align_items="start",
            spacing="2",
        ),

        rx.text(""),
        rx.heading("Enter your swing data to determine which cluster it identifies most closely with.",
                   margin_top="40px",),

        rx.hstack(
            rx.vstack(
                rx.text("Bat Speed (MPH)"),
                rx.input(
                    value = AppState.bat_speed_input,
                    on_change = AppState.set_bat_speed_input,
                    width = "100%",
            ),
            width = "25%",
        ),
        
        rx.vstack(
            rx.text("Attack Angle (deg)"),
            rx.input(
                value = AppState.attack_angle_input,
                on_change = AppState.set_attack_angle_input,
                width = "100%",
            ),
            width = "25%",
        ),

        rx.vstack(
            rx.text("Vertical Bat Angle (deg)"),
            rx.input(
                value = AppState.vba_input,
                on_change = AppState.set_vba_input,
                width = "100%",
            ),
            width = "25%",
        ),

        rx.vstack(
            rx.text("Time to Contact (s)"),
            rx.input(
                value = AppState.ttc_input,
                on_change = AppState.set_ttc_input,
                width = "100%",
            ),
            width = "25%",
        ),

        width = "100%",
        spacing = "5",
        ),

        rx.button(
            "Predict",
            on_click = AppState.predict_cluster,
        ),

        rx.cond(
            AppState.prediction_error != "",
            rx.text(
                AppState.prediction_error,
                color = "red",
            ),
        ),

        rx.cond(
            AppState.prediction_ready,
            rx.vstack(
                rx.text(f"🔴: {AppState.predicted_probs[0]}%"),
                rx.text(f"🔵: {AppState.predicted_probs[1]}%"),
                rx.text(f"🟢: {AppState.predicted_probs[2]}%"),
                rx.text(f"🟠: {AppState.predicted_probs[3]}%"),
                align_items="start",
            ),
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