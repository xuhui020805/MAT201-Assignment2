import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# --- 1. Page Configuration & Title ---
st.set_page_config(page_title="Gradient & Steepest Ascent Visualizer", layout="wide")

st.title("🏔️ Interactive Gradient & Steepest Ascent Visualizer")
st.markdown("""
**Calculus MAT201 Assignment 2** This application visually demonstrates the concepts of **Gradient Vectors** and the **Direction of Steepest Ascent** for functions of several variables $f(x, y)$.
""")

# --- 2. Sidebar: User Inputs ---
st.sidebar.header("⚙️ Settings")

# Function Selection
function_choice = st.sidebar.selectbox(
    "Choose a Function Surface:",
    (
        "10 - x**2 - y**2", 
        "x * exp(-(x**2 + y**2))", 
        "sin(x) * cos(y)"
    )
)

# Coordinate Input
st.sidebar.subheader("Adjust Point (x, y)")
x_val = st.sidebar.slider("x value", -3.0, 3.0, 1.0, 0.1)
y_val = st.sidebar.slider("y value", -3.0, 3.0, 1.0, 0.1)

# --- 3. Mathematical Calculations ---
x, y = sp.symbols('x y')

# Parse function string
if function_choice == "10 - x**2 - y**2":
    f_expr = 10 - x**2 - y**2
elif function_choice == "x * exp(-(x**2 + y**2))":
    f_expr = x * sp.exp(-(x**2 + y**2))
else:
    f_expr = sp.sin(x) * sp.cos(y)

# Calculate Partial Derivatives
fx = sp.diff(f_expr, x)
fy = sp.diff(f_expr, y)

# Convert to Python functions for numerical evaluation
f_func = sp.lambdify((x, y), f_expr, 'numpy')
fx_func = sp.lambdify((x, y), fx, 'numpy')
fy_func = sp.lambdify((x, y), fy, 'numpy')

# Calculate values at the current point
z_val = f_func(x_val, y_val)
grad_x = fx_func(x_val, y_val)
grad_y = fy_func(x_val, y_val)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# --- 4. Display Mathematical Results ---
col1, col2 = st.columns(2)

with col1:
    st.info("### 📐 Mathematical Concepts")
    st.latex(r"f(x, y) = " + sp.latex(f_expr))
    st.write(f"**Current Point:** $P({x_val}, {y_val})$")
    st.write(f"**Height ($z$):** {z_val:.4f}")

with col2:
    st.success("### 🚀 Gradient Results")
    st.write("The gradient vector $\\nabla f$ points in the direction of **steepest ascent**.")
    st.latex(r"\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle = \langle " + f"{grad_x:.3f}, {grad_y:.3f}" + r"\rangle")
    st.write(f"**Steepest Ascent Rate (Magnitude):** {magnitude:.4f}")

st.divider()

# --- 5. Visualization (3D Plot) ---
st.subheader("📊 3D Visualization & Gradient Direction")

# Generate Grid Data
x_range = np.linspace(-3, 3, 50)
y_range = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = f_func(X, Y)

# Create 3D Figure
fig = go.Figure()

# Plot Surface
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Surface'))

# Plot Current Point
fig.add_trace(go.Scatter3d(
    x=[x_val], y=[y_val], z=[z_val],
    mode='markers', marker=dict(size=8, color='red'), name='Current Point'
))

# Plot Gradient Vector (Arrow)
# We visualize the vector starting from the point on the surface
arrow_scale = 0.5
fig.add_trace(go.Scatter3d(
    x=[x_val, x_val + grad_x * arrow_scale],
    y=[y_val, y_val + grad_y * arrow_scale],
    z=[z_val, z_val + magnitude * arrow_scale * 0.5], # Visual approximation for 3D slope
    mode='lines+markers',
    line=dict(color='red', width=5),
    name='Steepest Ascent Direction'
))

fig.update_layout(
    title='3D Surface Plot (Interact: Drag to Rotate)',
    autosize=True,
    height=600,
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis (Function Value)'
    )
)
st.plotly_chart(fig, use_container_width=True)

# --- 6. Real World Application & Summary ---
st.divider()
st.subheader("🌍 Real World Application: Heat Seeking")
st.markdown("""
**Concept:** The Gradient Vector always points to the direction where the function increases the fastest.

**Real World Example:**
Imagine a **heat-seeking drone** or a **robot** trying to find the hottest source of fire in a room. 
* The function $f(x,y)$ represents the temperature distribution in the room.
* The robot continuously calculates the **Gradient** of the temperature at its current position.
* By moving in the direction of the gradient vector, the robot takes the path of **steepest ascent** (getting hotter as fast as possible) to locate the fire source efficiently.

**In this App:**
The **Red Line** in the 3D plot above shows exactly which direction you should step to climb the hill fastest!
""")

st.caption("Developed for MAT201 Assignment 2.")
