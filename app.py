import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# --- 1. Page Configuration & Title ---
st.set_page_config(page_title="Gradient Master Pro", layout="wide")

st.title("🏔️ Interactive Gradient & Steepest Ascent Visualizer")
st.markdown("""
**Calculus MAT201 Assignment 2**
This application visualizes the **Gradient Vector** and **Direction of Steepest Ascent** for any function of two variables $f(x, y)$.
""")

# --- 2. Sidebar: User Settings ---
st.sidebar.header("⚙️ Settings")

# Option to switch between Preset and Custom Input
input_mode = st.sidebar.radio(
    "Input Mode:",
    ("Select Preset Function", "Enter Custom Formula")
)

# Initialize variables
x, y = sp.symbols('x y')
f_expr = None
func_str = ""

# Logic for Input Mode
if input_mode == "Select Preset Function":
    # Dropdown for safe, pre-defined functions
    func_str = st.sidebar.selectbox(
        "Choose a Function:",
        (
            "10 - x^2 - y^2", 
            "x * exp(-(x^2 + y^2))", 
            "sin(x) * cos(y)",
            "x^2 - y^2"
        )
    )
else:
    # Text Input for custom formulas
    st.sidebar.info("💡 **Tip:** You can use `^` for power (e.g., `x^2`). Supported functions: `sin`, `cos`, `exp`, `sqrt`, `log`.")
    func_str = st.sidebar.text_input("Enter f(x, y):", value="x^2 + y^2")

# --- 3. Mathematical Processing ---
try:
    # 1. Pre-processing: Replace '^' with '**' for Python syntax compatibility
    # This allows users to type "x^2" instead of "x**2"
    clean_expr_str = func_str.replace("^", "**")
    
    # 2. Parse the string into a SymPy expression
    f_expr = sp.sympify(clean_expr_str)
    
    # 3. Calculate Partial Derivatives symbolically
    fx = sp.diff(f_expr, x)
    fy = sp.diff(f_expr, y)

    # 4. Create numerical functions (lambdify) for fast calculation
    f_func = sp.lambdify((x, y), f_expr, 'numpy')
    fx_func = sp.lambdify((x, y), fx, 'numpy')
    fy_func = sp.lambdify((x, y), fy, 'numpy')

    # 5. User Coordinate Input
    st.sidebar.subheader("Adjust Point P(x, y)")
    x_val = st.sidebar.slider("x coordinate", -3.0, 3.0, 1.0, 0.1)
    y_val = st.sidebar.slider("y coordinate", -3.0, 3.0, 1.0, 0.1)

    # 6. Calculate values for the specific point
    z_val = f_func(x_val, y_val)
    grad_x = fx_func(x_val, y_val)
    grad_y = fy_func(x_val, y_val)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # --- 4. Display Math Results ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 📐 Function & Point Analysis")
        # Try to display pretty LaTeX, fallback to string if complex
        st.latex(r"f(x, y) = " + sp.latex(f_expr))
        st.write(f"**Current Point:** $P({x_val}, {y_val})$")
        st.write(f"**Function Height ($z$):** {z_val:.4f}")

    with col2:
        st.success("### 🚀 Gradient Calculation")
        st.markdown("The **Gradient Vector** $\\nabla f$ points in the direction of steepest ascent.")
        st.latex(r"\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle")
        st.latex(r"\nabla f = \langle " + f"{grad_x:.3f}, {grad_y:.3f}" + r"\rangle")
        st.write(f"**Steepest Ascent Rate (Magnitude):** {magnitude:.4f}")

    st.divider()

    # --- 5. 3D Visualization Logic ---
    st.subheader("📊 3D Surface Visualization")

    # Generate Grid for plotting
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Evaluate Z over the grid
    # Using a nested try-except specifically for plotting errors (e.g. division by zero)
    try:
        Z = f_func(X, Y)
        
        # Initialize Plotly Figure
        fig = go.Figure()

        # Layer 1: The Surface
        fig.add_trace(go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis', 
            opacity=0.8, 
            name='Surface f(x,y)'
        ))
        
        # Layer 2: The Current Point
        fig.add_trace(go.Scatter3d(
            x=[x_val], y=[y_val], z=[z_val],
            mode='markers', 
            marker=dict(size=6, color='red', symbol='circle'), 
            name='Current Point P'
        ))

        # Layer 3: The Gradient Vector (Arrow)
        # We draw a line starting from the point P, extending in the direction of the gradient
        arrow_length = 0.5
        fig.add_trace(go.Scatter3d(
            x=[x_val, x_val + grad_x * arrow_length],
            y=[y_val, y_val + grad_y * arrow_length],
            # We visualize the Z-component direction proportional to magnitude for 3D effect
            z=[z_val, z_val + magnitude * arrow_length], 
            mode='lines+markers',
            line=dict(color='red', width=5),
            name='Gradient Vector (Steepest Ascent)'
        ))

        # Layout updates
        fig.update_layout(
            title='3D Terrain (Drag to Rotate)', 
            autosize=True, 
            height=600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as plot_err:
        st.warning(f"⚠️ Could not plot the surface for this function (Math Error: {plot_err}). Try adjusting the range or formula.")

except Exception as e:
    # Catch-all for invalid formulas (syntax errors)
    st.error(f"❌ **Formula Error:** We couldn't understand your function.")
    st.error(f"Error Details: {e}")
    st.info("💡 **Hint:** Make sure to use `*` for multiplication (e.g., `2*x` not `2x`).")

# --- 6. Real World Application Section ---
st.divider()
st.subheader("🌍 Real World Application: Gradient Descent & Ascent")
st.markdown("""
**Concept:** The Gradient Vector $\\nabla f$ always points in the direction of maximum increase.

**Application in AI (Gradient Descent):** In Machine Learning, algorithms use the *negative* gradient ($-\\nabla f$) to find the minimum point of a "Loss Function". This is like a ball rolling down into the deepest valley to minimize error.

**Application in Robotics (Gradient Ascent):**
A heat-seeking robot calculates the gradient of temperature to find the hottest point in a room. It follows the direction of steepest ascent, exactly as shown by the **red line** in the graph above.
""")

st.caption("Developed for Calculus MAT201 Assignment 2.")
