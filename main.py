import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'mujoco_menagerie/franka_emika_panda/scene.xml'

# Load the robot arm model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Find the body ID of the end-effector so we can track its Jacobian
# ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")

# Get the geom ID of the hand/gripper links to track collisions
# In MuJoCo, collisions happen between "geoms" (geometries), not bodies.
hand_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link7")

time_history = []
# force_z_history = []
# force_magnitude_history = []
contact_force_history = []

step_counter = 0
DOWNSAMPLE_FACTOR = 10 # Only save every 10th physics step (Tracks at 50Hz instead of 500Hz)

def custom_control_loop(model, data):
    """
    This function automatically executes at EVERY single physics time-step 
    (usually 1000Hz) inside the interactive viewer.
    """
    global step_counter

    # Scripted Trajectory (Simple continuous wave movement)
    # amplitude = 0.3
    # frequency = 0.5
    # target_velocity = amplitude * np.cos(2 * np.pi * frequency * data.time)
    
    # Let's override joint 1 (shoulder) to force the arm down toward the floor
    # Adjust this slider or control index depending on how your specific XML moves
    data.ctrl[1] = -1.5 if data.time < 3.0 else 0.0
    
    # Send velocity command to the first joint actuator as an example
    data.ctrl[0] = target_velocity 
    
    # Live Data Extraction 
    q = data.qpos[:7]            # Positions of the 7 joints
    qvel = data.qvel[:7]         # Velocities of the 7 joints
    tau = data.qfrc_actuator[:7] # Raw motor torques
    
    # Compute the 6x7 Jacobian Matrix for the hand's current position
    jac_p = np.zeros((3, model.nv)) # Jacobian Position
    jac_r = np.zeros((3, model.nv)) # Jacobian Rotation
    mujoco.mj_jacBody(model, data, jac_p, jac_r, ee_id)
    J = np.vstack([jac_p, jac_r])[:, :7]
    
    # Grab True External Contact Forces (Ground Truth)
    # MuJoCo tracks net external forces on bodies in data.cfrc_ext
    # Extract the external forces acting on our hand link
    true_ext_force_torque = data.cfrc_ext[ee_id] # 6D vector [torque_x, torque_y, torque_z, force_x, force_y, force_z]

    if step_counter % DOWNSAMPLE_FACTOR == 0:
    # Only save every 10th physics step (Tracks at 50Hz instead of 500Hz)
        force_x = true_ext_force_torque[3]
        force_y = true_ext_force_torque[4]
        force_z = true_ext_force_torque[5]
        
        total_force_mag = np.sqrt(force_x**2 + force_y**2 + force_z**2)

        time_history.append(data.time)
        force_z_history.append(force_z)
        force_magnitude_history.append(total_force_mag)

    step_counter += 1

    # Print a clean, readable stream to the terminal for verification
    print(f"Time: {data.time:.2f}s | Joint 1 Pos: {q[0]:.2f} | Net Ext Force Z: {true_ext_force_torque[5]:.2f} N")

mujoco.set_mjcb_control(custom_control_loop)

print("Launching interactive viewer. Click and drag the hand link (Ctrl + Two-finger drag) to create force spikes!")
print("Close the MuJoCo GUI window when you are done to generate the graph.")

mujoco.viewer.launch(model, data)

print("\nSimulation ended. Generating your timeline force graph...")

# Visualise force against time
t = np.array(time_history)
f_z = np.array(force_z_history)
f_mag = np.array(force_magnitude_history)

# Create the plot canvas
plt.figure(figsize=(10, 5))

# Plot total force magnitude as a solid blue line
plt.plot(t, f_mag, label='Total External Force Magnitude (N)', color='#1f77b4', linewidth=2)
# Plot localized Z-axis force as a dashed orange line to see directionality
plt.plot(t, f_z, label='Z-Axis Directional Force (N)', color='#ff7f0e', linestyle='--', alpha=0.8)

# Formatting polish
plt.title('End-Effector External Contact Force Timeline', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Time (Seconds)', fontsize=12)
plt.ylabel('Measured Force (Newtons)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)

# Display the final plot window
plt.tight_layout()
plt.show()