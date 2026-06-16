import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'mujoco_menagerie/franka_emika_panda/scene.xml'

# Load the robot arm model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Get the geom ID of the hand/gripper links to track collisions
# In MuJoCo, collisions happen between "geoms" (geometries), not bodies.
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
hand_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link7")

time_history = []
contact_force_history = []

step_counter = 0
DOWNSAMPLE_FACTOR = 10 # Only save every 10th physics step (Tracks at 50Hz instead of 500Hz)

def custom_control_loop(model, data):
    """
    This function automatically executes at EVERY single physics time-step 
    (usually 1000Hz) inside the interactive viewer.
    """
    global step_counter

    # Scripted trajectory
    data.ctrl[1] = -1.0
    data.ctrl[2] = 0.5

    if step_counter % DOWNSAMPLE_FACTOR == 0:
    # Only save every 10th physics step (Tracks at 50Hz instead of 500Hz)
        total_normal_force = 0.0

        # Loop through every active contact collision currently in the simulator
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Identify which structural bodies own the two colliding geometries
            body1_id = model.geom_bodyid[contact.geom1]
            body2_id = model.geom_bodyid[contact.geom2]
            
            # If either body matching the collision is our hand link (ee_id)
            if body1_id == ee_id or body2_id == ee_id:
                # Extract the raw contact force vector
                c_forces = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, c_forces)
                
                # c_forces[0] is the normal force pushing perpendicular to the floor
                total_normal_force += c_forces[0]
        
        # Log data
        time_history.append(data.time)
        contact_force_history.append(total_normal_force)

    step_counter += 1

    # Print a clean, readable stream to the terminal for verification
    # print(f"Time: {data.time:.2f}s | Net Ext Force: {total_normal_force:.2f} N")

mujoco.set_mjcb_control(custom_control_loop)

print("Launching Collision-Aware Viewer...")
print("Drive the arm into the floor using the 'Control' sliders on the right panel, or let the script run.")

mujoco.viewer.launch(model, data)

print("\nSimulation ended. Generating your timeline force graph...")

# Visualise force against time
t = np.array(time_history)
f_contact = np.array(contact_force_history)

plt.figure(figsize=(10, 5))
plt.plot(t, f_contact, label='True Normal Contact Force (N)', color='#d62728', linewidth=2)

plt.title('End-Effector Floor Collision Impact Profile', fontsize=14, fontweight='bold')
plt.xlabel('Simulation Time (Seconds)', fontsize=12)
plt.ylabel('Impact Force (Newtons)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()