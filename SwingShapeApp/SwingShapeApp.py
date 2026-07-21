import reflex as rx

### App state class
class AppState(rx.State):
    current_tab: str = "Landing"

    def set_landing(self):
        self.current_tab = "Landing"
    
    def set_model_visual(self):
        self.current_tab = "Model Visuals"

    def set_swingshape(self):
        self.current_tab = "Swing Shape"
    
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
    return rx.vstack(rx.heading("Swing Shape"), rx.text("Swing Shape"), align_items = "start", )


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