import pybounds
import numpy as np
import matplotlib.pyplot as plt 
params = dict(m=20.0, kv=300.0, bv=20.0,
              Iz=1.0, kw=50.0, bw=2.0)

def f(x, u, m=params['m'], kv=params['kv'], bv=params['bv'], Iz=params['Iz'], kw=params['kw'], bw=params['bw']):
    """
    UGV dynamics in continuous time for odeint.
    """
    v_cmd, w_cmd = u

    # Unpack state
    X, Y, theta, xdot, ydot, thetadot = x

    # Body-frame forward velocity
    v_body = np.cos(theta) * xdot + np.sin(theta) * ydot

    # Effective acceleration coefficients
    alpha_v = kv / m
    gamma_v = (kv + bv) / m
    alpha_w = kw / Iz
    gamma_w = (kw + bw) / Iz

    # Accelerations
    xddot = -gamma_v * v_body * np.cos(theta) + alpha_v * np.cos(theta) * v_cmd
    yddot = -gamma_v * v_body * np.sin(theta) + alpha_v * np.sin(theta) * v_cmd
    thetaddot = -gamma_w * thetadot + alpha_w * w_cmd

    # Collect derivatives
    return [
        xdot,       # dx/dt
        ydot,       # dy/dt
        thetadot,   # dtheta/dt
        xddot,      # d(xdot)/dt
        yddot,      # d(ydot)/dt
        thetaddot   # d(thetadot)/dt
    ]


def h(x_vec, u_vec):
    # Extract state variables
    X = x_vec[0]
    Y = x_vec[1]
    theta = x_vec[2]
    xdot = x_vec[3]
    ydot = x_vec[4]
    thetadot = x_vec[5]

    # Extract control inputs
    v = u_vec[0]
    w = u_vec[1]

    #Add gaussian noise to measurements
    X_new = X + np.random.normal(0, 0.2)
    Y_new = Y + np.random.normal(0, 0.2)
    theta_new = theta + np.random.normal(0, 0.1)
    # Measurements
    y_vec = [X_new, Y_new, theta_new]  # position and heading

    # Return measurement
    return y_vec

state_names = ['X', 'Y', 'theta', 'xdot', 'ydot', 'thetadot']
input_names = ['v', 'w']
measurement_names = ['X', 'Y', 'theta']
dt = 0.1

tsim = np.arange(0, 8.0, step=dt)


simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names, 
                               input_names=input_names, measurement_names=measurement_names, mpc_horizon=10)
NA = np.zeros_like(tsim)
setpoint = {'X': tsim,  # move along X axis at 1 m/s
            'Y': np.sin(tsim),  # oscillate in Y
            'theta': NA,
            'xdot': NA,
            'ydot': NA,
            'thetadot': NA,
           }
simulator.update_dict(setpoint, name='setpoint')

cost_x = (simulator.model.x['X'] - simulator.model.tvp['X_set']) ** 2
cost_z = (simulator.model.x['Y'] - simulator.model.tvp['Y_set']) ** 2

cost = cost_x + cost_z

simulator.mpc.set_objective(mterm=cost, lterm=cost)  # objective function

# Set input penalty: make this small for accurate state tracking
simulator.mpc.set_rterm(v=1e-4, w=1e-4)
simulator.mpc.bounds['upper', '_u', 'v'] = 1.5
simulator.mpc.bounds['lower', '_u', 'v'] = -1.5
simulator.mpc.bounds['upper', '_u', 'w'] = 0.8
simulator.mpc.bounds['lower', '_u', 'w'] = -0.8
t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, u=None, mpc=True, return_full_output=True)


#Plot trajectory
plt.figure()
plt.plot(x_sim['X'], x_sim['Y'], label='Trajectory')
plt.scatter(y_sim['X'], y_sim['Y'], color='red')
plt.plot(tsim, np.sin(tsim), 'r--', label='Setpoint Y=sin(t)', color='green')
plt.title('Trajectory in XY-plane')
simulator.plot('setpoint')
plt.show()